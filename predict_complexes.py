"""Score arbitrary protein-ligand complexes with a trained SS-GNN checkpoint.

Unlike evaluate.py this needs no ground-truth labels, so it works on structures
that have no measured affinity — e.g. individual frames of an MD trajectory.

Input is a directory of PDBbind-style complex directories:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf     (or _ligand.mol2)

usage:
    uv run python predict_complexes.py --complexes ~/md_data/frames \
        --model best_models/model_111.pt --out preds.csv
"""

import argparse
import csv
import os
from multiprocessing import Pool

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from tqdm import tqdm

from dataloader import DataLoader
from model import GINNet
from utils import get_gnn_features, set_data_device

RDLogger.DisableLog('rdApp.*')


def load_complex(complex_dir):
    """Read protein and ligand from one complex directory."""
    cid = os.path.basename(complex_dir.rstrip('/'))

    protein = Chem.MolFromPDBFile(os.path.join(complex_dir, cid + '_protein.pdb'))
    if protein is None:
        pocket = os.path.join(complex_dir, cid + '_pocket.pdb')
        if os.path.exists(pocket):
            protein = Chem.MolFromPDBFile(pocket)

    ligand = None
    mol2 = os.path.join(complex_dir, cid + '_ligand.mol2')
    if os.path.exists(mol2):
        ligand = Chem.MolFromMol2File(mol2)
    if ligand is None:
        sdf = os.path.join(complex_dir, cid + '_ligand.sdf')
        if os.path.exists(sdf):
            mols = [m for m in Chem.SDMolSupplier(sdf) if m]
            ligand = mols[0] if mols else None

    return cid, protein, ligand


def featurize(args):
    complex_dir, threshold = args
    cid, protein, ligand = load_complex(complex_dir)
    if protein is None or ligand is None:
        return cid, None
    try:
        x, edge_index, edge_attr = get_gnn_features(protein, ligand, threshhold=threshold)
        if x is None or edge_index is None or edge_attr is None:
            return cid, None
        return cid, (x, edge_index, edge_attr)
    except Exception:
        return cid, None


class ComplexDataset(torch.utils.data.Dataset):
    """Graphs only — the label slot is a placeholder so the collater stays happy."""

    def __init__(self, graphs):
        self.ids = [cid for cid, g in graphs]
        self.graphs = [g for cid, g in graphs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x, edge_index, edge_attr = self.graphs[idx]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=x[:, -3:])
        return data, torch.tensor(0.0, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser(description='Predict binding affinity for arbitrary complexes')
    parser.add_argument('--complexes', required=True,
                        help='Directory of PDBbind-style complex directories')
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--out', default=None, help='Optional: save predictions to this .csv')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Distance cutoff in Angstroms (default: 5.0)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    dirs = sorted(
        os.path.join(args.complexes, d) for d in os.listdir(args.complexes)
        if os.path.isdir(os.path.join(args.complexes, d))
    )
    print(f'Featurizing {len(dirs)} complexes with {args.workers} workers (threshold={args.threshold}A)')

    tasks = [(d, args.threshold) for d in dirs]
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(featurize, tasks), total=len(tasks)))

    graphs = [(cid, g) for cid, g in results if g is not None]
    failed = [cid for cid, g in results if g is None]
    if failed:
        print(f'Skipped {len(failed)} complexes that could not be featurized: '
              f'{failed[:10]}{"..." if len(failed) > 10 else ""}')
    if not graphs:
        print('Nothing to predict.')
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = GINNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    dataset = ComplexDataset(graphs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc='Predicting', leave=False):
            x = set_data_device(x, device)
            preds.extend(model(x).squeeze(1).cpu().numpy().tolist())

    print(f'\n=== Predicted -log(Kd/Ki) for {len(preds)} complexes ===')
    for cid, p in zip(dataset.ids, preds):
        print(f'{cid:40s} {p:.3f}')

    if args.out:
        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['complex_id', 'y_pred'])
            writer.writerows(zip(dataset.ids, preds))
        print(f'\nPredictions saved to {args.out}')


if __name__ == '__main__':
    main()
