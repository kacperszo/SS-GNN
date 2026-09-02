# CLAUDE.md — SS-GNN

## What this is

Affinity prediction from a protein–ligand interaction graph. Fork of the authors' repo,
reproduced for the gnn-benchmark harness. Runs as `ssgnn.modern` there.

## Current state

Reproduction gate **green** as of 2026-08-31: fresh featurisation through the harness
scores RMSE=1.4173, R=0.7793 on the CASF-2016 core set with `best_models/model_665.pt`,
matching the stored `data/processed/coreset_graphs` reference to four decimals.

Container image builds and gives predictions bit-identical to the venv on CPU. GPU works
through rootless Podman with CDI; the full 285-complex core set takes 38 s.

## Hard-won facts (do NOT regress these)

- **The published preprocessing script is not the one that produced the paper.**
  `gnn_features.py` in the public repo omits covalent bonds between pocket atoms.
  `gnn_features_pEdge.py`, from the supporting-information ZIP and not on GitHub, adds
  them. Without them CASF-2016 scores **R=0.677 instead of 0.779** — a 0.10 gap that looks
  like a bad model and is actually a missing edge type. The block is now ported into
  `utils.get_gnn_features`.
- **`edge_attr` is cast to `torch.long` on purpose.** It truncates the distance column to
  integer ångströms, which looks like a bug and is not — the authors' own script does it,
  and the checkpoints were trained that way. Reverting it to float changes nothing on its
  own (still 0.677 without the pocket bonds), so do not "fix" it.
- **The authors load the protein from a pickled RDKit mol**, not from a PDB:
  `gnn_features_pEdge.py:174` is `protein = pickle.load(f)`. That pickle comes from an
  internal pipeline ("viewpro") that was never released, so bond and atom perception in
  the original differs from RDKit's PDB parser in ways we cannot reconstruct. This is the
  one remaining gap between us and the paper.
- **The paper's numbers do not reproduce.** Reported: best R=0.870, mean 0.853 ± 0.012 over
  5 seeds. Measured here: mean **0.786 ± 0.0115**, best of five runs 0.8062. The gap is
  systematic, not initialisation variance. Either the viewpro features explain it or the
  reported figures were selected.
- **Training has no fixed seed.** A single run on 2026-03-31 reached R=0.8128
  (`model_527.pt`), and that checkpoint no longer exists — the 2026-04-01 run overwrote
  `best_models/`. That result is unreproducible in principle. Set a seed before any number
  from here is quoted.
- **Everything in `best_models/` is from 2026-04-01, 15:00–16:33**, trained on graphs built
  at 14:59 with the pEdge corrections. Commit 82f15ea at 17:37 captured only the `long`
  cast; the pocket-bond change was never committed until 2026-08-31, which is why fresh
  featurisation silently disagreed with the stored graphs for months.
- **Concordance index depends on whose implementation you use.** On identical predictions:
  `lifelines(y_true, y_pred)`=0.7957, `lifelines(y_pred, y_true)`=0.7951, the harness's own
  =0.7876. When comparing against a published CI, the paper's implementation matters.
- **`torch_sparse` and `torch_scatter` have no wheels for torch 2.11** (data.pyg.org stops
  at 2.9). The ones in `wheels/` were compiled locally against Python 3.13 + torch 2.11.0.
  Change either and they stop importing, which is why the Containerfile pins both.
- **`.containerignore` has no inline comments.** `#` is only honoured at the start of a
  line; `data/  # comment` silently becomes a literal pattern and 988 MB of processed
  graphs end up in the image.

## Build & run

```bash
podman build --format=docker -t ssgnn:latest .        # micromamba-free, python:3.13-slim base
gnnb run --variant ssgnn.modern --capability predict --dataset <complexes> --gpu
```

`predict_complexes.py` is ours, not the authors': it scores complexes with no ground-truth
labels, which `evaluate.py` cannot do.

## Embedding

Done, and it required inventing a pooling. SS-GNN has **no graph-level vector**: `self.linear`
(256→512→512→256→1) runs per edge and `global_add_pool` sums scalars straight into the
prediction. `embed_complexes.py` hooks the input to `self.linear` and pools per complex — 256
dims, recorded as `defined`.

**Probe R 0.694**, retaining 89% of the model's own Pearson R, against the size baseline's 0.470.

That 89% is the same figure PLANET reaches, and both sit below every model where the authors'
own aggregation was available — IGN's native 200-dim vector retains 95%, and GenScore's
distance-masked sum reaches 0.796 outright. The pooling choice is part of the method, not an
implementation detail.
- Scaffold split, to see what the model does on chemically distinct structures.
- Hyperparameter retuning on properly defined splits.
