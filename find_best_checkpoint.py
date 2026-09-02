"""Evaluate all checkpoints in best_models/ on CASF-2016 and print a sorted table."""
import os
import pickle

import torch
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from dataloader import DataLoader
from evaluate import CoresetDataset, load_coreset_labels
from model import GINNet
from utils import set_data_device

GRAPHS = 'data/processed/coreset_graphs'
LABELS = 'data/CASF-2016/power_scoring/CoreSet.dat'
MODELS_DIR = 'best_models'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_checkpoint(path):
    model = GINNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = set_data_device((x, y), DEVICE)
            score = model(x)
            y_pred.extend(score.squeeze(1).cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())
    r = pearsonr(y_true, y_pred)[0]
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    ci = concordance_index(y_true, y_pred)
    return r, rmse, ci


labels = load_coreset_labels(LABELS)
dataset = CoresetDataset(GRAPHS, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

def parse_epoch(fname):
    # handles model_527.pt and model_527_run1.pt
    stem = fname.replace('model_', '').replace('.pt', '')
    return int(stem.split('_')[0])

checkpoints = sorted(
    [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')],
    key=parse_epoch
)

results = []
for fname in checkpoints:
    path = os.path.join(MODELS_DIR, fname)
    r, rmse, ci = eval_checkpoint(path)
    results.append((fname, r, rmse, ci))
    print(f'{fname:30s} | R={r:.4f} | RMSE={rmse:.4f} | CI={ci:.4f}')

results.sort(key=lambda x: x[1], reverse=True)
print('\n=== Top 5 by Pearson R (global) ===')
for fname, r, rmse, ci in results[:5]:
    print(f'  {fname}  R={r:.4f}  RMSE={rmse:.4f}  CI={ci:.4f}')

# Best per run
from collections import defaultdict
best_per_run = defaultdict(lambda: ('', -1, 0, 0))
for fname, r, rmse, ci in results:
    run = fname.split('run')[1].split('.')[0] if 'run' in fname else '0'
    if r > best_per_run[run][1]:
        best_per_run[run] = (fname, r, rmse, ci)

print('\n=== Best per run ===')
rs = []
for run in sorted(best_per_run):
    fname, r, rmse, ci = best_per_run[run]
    print(f'  run{run}: {fname:30s}  R={r:.4f}  RMSE={rmse:.4f}  CI={ci:.4f}')
    rs.append(r)

if len(rs) > 1:
    import numpy as np
    print(f'\n  Mean R={np.mean(rs):.4f} ± {np.std(rs):.4f}  '
          f'(paper: 0.853 ± 0.012)')
