# SS-GNN — a deliberately small joint graph over ligand and pocket

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

Two GINConv layers over one joint ligand-pocket graph, then a per-edge MLP whose
scalar outputs are summed into the prediction. The paper's point is that a shallow network on a
well-chosen graph competes with much larger ones.

**Atom coordinates enter as node features.** `utils.py:153,158` append each atom's raw xyz to its
feature vector and `model.py:81` concatenates `data.pos / 10` into the node representation — the
authors' own design, and the reason this model is not invariant to where the complex sits.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.779**, RMSE 1.417, n=285 |
| embedding | 256d, **ours**, probe R 0.694 — 89% of its own head |
| `gnnb verify` | 285/285, 5.4e-05 |
| invariance | **none declared** — a translation that changes no distance moves the prediction 0.48 |

## Build

```bash
podman build --format=docker -t ssgnn:latest .   # python:3.13-slim, torch 2.11
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

A joint ligand-pocket graph built from the protein and ligand files directly.

```bash
# ssgnn.modern — localhost/ssgnn:latest
# source: models/ssgnn

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/best_models:/ckpt:ro" \
    localhost/ssgnn:latest \
    python predict_complexes.py --complexes /data --model /ckpt/model_665.pt --out /outputs/predictions.csv --device cpu

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/best_models:/ckpt:ro" \
    localhost/ssgnn:latest \
    python embed_complexes.py --complexes /data --model /ckpt/model_665.pt --out /outputs/embeddings.npz --pool sum --device cpu
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` |
| `embeddings.npz` | `ids` and `vectors`, 256-dim — ours. `self.linear` runs *per edge* and `global_add_pool` sums its scalars, so there is no graph-level vector to strip a head from; we pool the head's input instead |

## Before you trust the numbers

**No compiled PyG extensions.** `torch_sparse` was required by one import in `batch.py`, a vendored copy of PyG's old `Batch` whose SparseTensor branches are dead code here — SS-GNN has no `adj_t` and no `ToSparseTensor`. That import is optional now, which is what lets this image track a current torch; before, it needed wheels compiled locally against one exact torch version, and those wheels were gitignored, so the image could not be rebuilt from a clone at all.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
