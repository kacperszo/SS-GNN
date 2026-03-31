## SS-GNN

> This is a Pytorch implementation of `SS-GNN`, a simple-structured GNN model for drug-target binding affinity (DTBA) prediction as described in the following paper:


The `SS-GNN` defines the prediction of DTBA as a regression task, in which the model’s input is the drug-target representation, and the output is a continuous value representing the binding affinity score between the drug and the target protein. The overall architecture of the `SS-GNN` is shown in the figure below.

![pic1](./images/1.jpg)


## Get Started

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

### 2. Download data

Download [PDBbind v2019](http://www.pdbbind.org.cn/) and [CASF-2016](http://www.pdbbind.org.cn/casf.php) and place them under `data/`:

```
data/
  v2019/          # PDBbind general set — one subdirectory per complex
  CASF-2016/
    coreset/      # 285 complexes used as test set
```

### 3. Preprocess

Build binding affinity labels:

```bash
uv run python make_labels.py
```

Extract graph features from raw PDB/mol2 files (resumable — safe to interrupt and re-run):

```bash
uv run python gnn_features.py
```

Options:

```
--pdbbind   Path to PDBbind directory       (default: data/v2019)
--coreset   Path to CASF core set directory (default: data/CASF-2016/coreset)
--out       Output directory for graphs     (default: data/processed/graphs)
--threshold Distance cutoff in Ångströms    (default: 5.0)
--workers   Number of parallel workers      (default: cpu_count - 1)
```

After preprocessing, `data/processed/` will contain:

```
data/processed/
  graphs/       # one .pkl per complex: (x, edge_index, edge_attr)
  labels.pkl    # dict {pdb_id: -log(Kd/Ki)}
```

### 4. Train

```bash
uv run python train.py
```

Options:

```
--data    Path to processed graphs directory  (default: data/processed/graphs)
--labels  Path to labels pickle               (default: data/processed/labels.pkl)
--device  Torch device                        (default: cuda:0)
--runs    Number of independent runs          (default: 1)
```

To reproduce the paper's multi-run average (Table 5), use `--runs 5` or more. Each run saves its model as `saved_models/model_run0.pt`, `model_run1.pt`, etc.

### 5. Evaluate on CASF-2016

Preprocess the core set (285 complexes, one-time):

```bash
uv run python make_coreset_graphs.py
```

Run evaluation on a saved checkpoint:

```bash
uv run python evaluate.py --model best_models/model_42.pt
```

Options:

```
--model       Path to model checkpoint (.pt)          (required)
--graphs      Path to coreset graphs                  (default: data/processed/coreset_graphs)
--labels      Path to CoreSet.dat                     (default: data/CASF-2016/power_scoring/CoreSet.dat)
--device      Torch device                            (default: cuda:0)
--batch-size  Batch size                              (default: 32)
--out         Save predictions to .csv                (optional)
```


## Known data issues

- **`6fuk` missing label**: In PDBbind v2019, the structural data directory is named `6fuk` but the index file (`INDEX_general_PL_data.2019`) records it as `6ful`. This is a typo in the original database. As a result, `6fuk` is silently skipped during training (the dataset filters to complexes that have both a graph and a label).

## Citation

When using this project in your research, please cite:
<section id="citation">
  <blockquote>
    Zhang, S., Jin, Y., Liu, T., Wang, Q., Zhang, Z., Zhao, S., & Shan, B. (2023).<br>
    <strong>SS-GNN: A Simple-Structured Graph Neural Network for Affinity Prediction.</strong><br>
    <i>ACS Omega</i>, 8(25), 22496–22507.<br>
    <a href="https://doi.org/10.1021/acsomega.3c00085">https://doi.org/10.1021/acsomega.3c00085</a>
  </blockquote>
</section>
