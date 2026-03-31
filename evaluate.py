import argparse
import os
import pickle

import numpy as np
import torch
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data
from tqdm import tqdm

from dataloader import DataLoader
from model import GINNet
from utils import set_data_device


class CoresetDataset(torch.utils.data.Dataset):
    def __init__(self, graphs_dir, labels):
        self.graphs_dir = graphs_dir
        self.labels = labels
        self.ids = [f.split('.')[0] for f in os.listdir(graphs_dir)
                    if f.endswith('.pkl') and f.split('.')[0] in labels]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pdb_id = self.ids[idx]
        with open(os.path.join(self.graphs_dir, pdb_id + '.pkl'), 'rb') as f:
            x, edge_index, edge_attr = pickle.load(f)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=x[:, -3:])
        label = torch.tensor(self.labels[pdb_id], dtype=torch.float)
        return data, label


def load_coreset_labels(coreset_dat):
    labels = {}
    with open(coreset_dat) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            labels[parts[0]] = float(parts[3])
    return labels


def evaluate(model, dataloader, device):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Evaluating', leave=False):
            x, y = set_data_device((x, y), device)
            score = model(x)
            y_pred.extend(score.squeeze(1).cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())

    mse = mean_squared_error(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    return mse, r, p, ci, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate SS-GNN on CASF-2016 core set')
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--graphs', default='data/processed/coreset_graphs',
                        help='Path to preprocessed coreset graphs (default: data/processed/coreset_graphs)')
    parser.add_argument('--labels', default='data/CASF-2016/power_scoring/CoreSet.dat',
                        help='Path to CoreSet.dat with binding affinities')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', default=None,
                        help='Optional: save predictions to this .csv file')
    args = parser.parse_args()

    if not os.path.exists(args.graphs) or len(os.listdir(args.graphs)) == 0:
        print(f'No graphs found at {args.graphs}.')
        print('Run: uv run python make_coreset_graphs.py')
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    labels = load_coreset_labels(args.labels)
    dataset = CoresetDataset(args.graphs, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = GINNet().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    mse, r, p, ci, y_true, y_pred = evaluate(model, dataloader, device)

    print(f'\n=== CASF-2016 Core Set Results ({len(dataset)} complexes) ===')
    print(f'MSE:     {mse:.4f}')
    print(f'RMSE:    {mse**0.5:.4f}')
    print(f'Pearson: {r:.4f}  (p={p:.2e})')
    print(f'CI:      {ci:.4f}')

    if args.out:
        import csv
        ids = dataset.ids
        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pdb_id', 'y_true', 'y_pred'])
            for pdb_id, yt, yp in zip(ids, y_true, y_pred):
                writer.writerow([pdb_id, yt, yp])
        print(f'Predictions saved to {args.out}')


if __name__ == '__main__':
    main()
