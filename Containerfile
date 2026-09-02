# SS-GNN, modern tier. CPU or GPU — the torch wheel carries CUDA either way.
#
# Python 3.13 and torch 2.11.0 are not free choices: torch_sparse and torch_scatter have
# no published wheels for torch 2.11 (data.pyg.org stops at 2.9), so the ones in wheels/
# were compiled locally against exactly this pair. Change either and they stop importing.
FROM docker.io/library/python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# torch first and alone: the prebuilt extensions below link against its ABI
RUN pip install --no-cache-dir torch==2.11.0

# locally built, because no index publishes them for this torch
COPY wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

RUN pip install --no-cache-dir \
    torch-geometric==2.7.0 \
    rdkit==2025.9.6 \
    numpy \
    tqdm

# verify every module the predict path pulls in, so a broken image fails at build time
RUN python -c "import torch, torch_geometric, torch_sparse, torch_scatter, rdkit; \
from torch_sparse import SparseTensor; \
print('env ok', torch.__version__)"

WORKDIR /work
COPY . /work
RUN python -c "import model, utils, batch, dataloader; print('ss-gnn imports ok')"
