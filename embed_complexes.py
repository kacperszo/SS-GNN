"""Extract a complex-level embedding from SS-GNN, with the prediction head removed.

There is no graph-level vector in this architecture to extract. The tail is

    xe  = torch.cat((x, e), 1)          # 256 per EDGE
    xe  = self.linear(xe)               # 256->512->512->256->1, still per edge
    out = global_add_pool(xe, edge_batch)   # sums SCALARS into the prediction

so removing the head leaves one 256-d vector per edge, not one per complex. A pooling the
authors never had has to be introduced. That makes this embedding *defined* rather than
*native*, and any cross-model comparison has to be read with that in mind.

What we take is the input to `self.linear` — the concatenation of the two GIN-convolved
node representations with the edge features, which is the last thing the network computes
before it starts collapsing towards a scalar.

usage:
    python embed_complexes.py --complexes /data --model /ckpt/model.pt \
        --out /outputs/embeddings.npz --pool sum
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from tqdm import tqdm

from dataloader import DataLoader
from model import GINNet
from predict_complexes import ComplexDataset, featurize
from utils import set_data_device

POOLING = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed complexes with the head removed")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default="sum", choices=sorted(POOLING))
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    dirs = sorted(
        os.path.join(args.complexes, d) for d in os.listdir(args.complexes)
        if os.path.isdir(os.path.join(args.complexes, d))
    )
    print(f"Featurizing {len(dirs)} complexes with {args.workers} workers")
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(featurize, [(d, args.threshold) for d in dirs]),
                            total=len(dirs)))

    graphs = [(cid, g) for cid, g in results if g is not None]
    skipped = [cid for cid, g in results if g is None]
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    if not graphs:
        print("Nothing to embed.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GINNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    # A forward pre-hook on the head captures its input, which is what we actually want.
    # Reaching in this way leaves the authors' forward() untouched, so the same checkpoint
    # keeps producing the same predictions.
    captured: list[torch.Tensor] = []
    handle = model.linear.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0])
    )

    dataset = ComplexDataset(graphs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pool_fn = POOLING[args.pool]

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Embedding", leave=False):
            captured.clear()
            x = set_data_device(x, device)
            model(x)
            pooled = pool_fn(captured[0], x.edge_batch)  # per edge -> per complex
            vectors.append(pooled.cpu().numpy())
    handle.remove()

    matrix = np.concatenate(vectors, axis=0)
    ids = np.array(dataset.ids)
    if len(ids) != len(matrix):
        raise RuntimeError(f"{len(ids)} ids but {len(matrix)} vectors")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, ids=ids, vectors=matrix)
    print(f"\n{len(ids)} embeddings of dimension {matrix.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
