# SS-GNN, modern tier. CPU or GPU — the torch wheel carries CUDA either way.
#
# **This image needs no compiled PyG extensions.** It used to: torch_sparse and torch_scatter
# have no published wheels for torch 2.11 (data.pyg.org stops at 2.9), so `wheels/` held copies
# built locally against exactly this torch — and `wheels/` is gitignored, which meant the image
# could not be rebuilt from a clone at all. The only thing that actually needed them was one
# import in `batch.py`, a vendored copy of PyG's old `Batch` whose SparseTensor branches are
# dead code here: SS-GNN has no `adj_t` and no `ToSparseTensor`. That import is optional now,
# so the dependency, the local build and the unreproducible image all go together.
FROM docker.io/library/python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torch-geometric==2.7.0 \
    rdkit==2025.9.6 \
    numpy \
    tqdm

# verify every module the predict path pulls in, so a broken image fails at build time
RUN python -c "import torch, torch_geometric, rdkit; \
import importlib.util as u; \
assert u.find_spec('torch_sparse') is None, 'torch_sparse crept back in'; \
assert u.find_spec('torch_scatter') is None, 'torch_scatter crept back in'; \
print('env ok', torch.__version__, 'pyg', torch_geometric.__version__)"

WORKDIR /work
COPY . /work
RUN python -c "import model, utils, batch, dataloader; \
import batch as b; assert b.SPARSE_TYPES == (), 'torch_sparse is present after all'; \
print('ss-gnn imports ok, no compiled extensions')"
