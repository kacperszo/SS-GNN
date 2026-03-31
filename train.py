# Copyright (C) 2022-2025 Shijiazhuang Xianyu Digital Biotechnology Co., Ltd

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
import time
import copy

import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

from model import GINNet as DeepMice
from dataset import get_gin_dataloader
from utils import set_data_device

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer:
    '''
            Drug Target Binding Affinity
    '''

    def __init__(self, data_path, label_path, device='cuda:0'):
        self.lr = 0.001
        self.decay = 0.00001
        self.BATCH_SIZE = 64
        self.train_epoch = 1000

        self.path = data_path
        self.label_path = label_path

        self.device = device
        self.config = None

        self.model = DeepMice()
        self.model = self.model.to(device)
        self.trainloader = get_gin_dataloader(self.path,
                                              self.label_path,
                                              self.BATCH_SIZE, phase='train')
        self.testloader = get_gin_dataloader(self.path,
                                             self.label_path,
                                             self.BATCH_SIZE,
                                             phase='test')
        self.valloader = get_gin_dataloader(self.path,
                                            self.label_path,
                                            self.BATCH_SIZE,
                                            phase='val')
        self.loss_fct = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.lr)

    def test_(self):
        y_pred = []
        y_label = []
        self.model.eval()
        for i, (x, y) in enumerate(self.testloader):
            x, y = set_data_device(
                (x, y), self.device)
            score = self.model(x)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        self.model.train()
        return [mean_squared_error(y_label, y_pred),
                pearsonr(y_label, y_pred)[0],
                pearsonr(y_label, y_pred)[1],
                concordance_index(y_label, y_pred), y_pred]

    def val_(self):
        y_pred = []
        y_label = []
        self.model.eval()
        for i, (x, y) in enumerate(self.valloader):
            x, y = set_data_device(
                (x, y), self.device)
            score = self.model(x)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        self.model.train()
        return [mean_squared_error(y_label, y_pred),
                pearsonr(y_label, y_pred)[0],
                pearsonr(y_label, y_pred)[1],
                concordance_index(y_label, y_pred), y_pred]

    def train(self):
        max_MSE = 10000
        model_max = copy.deepcopy(self.model)
        print('--- Go for Training ---')
        t_start = time.time()
        currentPearson = 0.0
        epoch_bar = tqdm(range(self.train_epoch), desc='Epochs')
        for epo in epoch_bar:
            batch_bar = tqdm(self.trainloader, desc=f'Epoch {epo+1}', leave=False)
            for i, (x, y) in enumerate(batch_bar):
                x, y = set_data_device(
                    (x, y), self.device)
                score = self.model(x)
                loss = self.loss_fct(score.squeeze(1), y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % 10 == 0:
                    batch_bar.set_postfix(loss=f'{loss.item():.4f}')

            # validate, select the best model up to now
            with torch.set_grad_enabled(False):
                mse, r2, p_val, CI, logits = self.val_()
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                epoch_bar.set_postfix(MSE=f'{mse:.4f}', R=f'{r2:.4f}', CI=f'{CI:.4f}')
                print(f'Epoch {epo+1} | MSE: {mse:.4f} | Pearson: {r2:.4f} '
                      f'(p={p_val:.2e}) | CI: {CI:.4f}')
            self.save_model('saved_models/model_'+str(epo)+'.pt')
            if r2 > currentPearson:
                self.save_model('best_models/model_' + str(epo) + '.pt')
                currentPearson = r2
        self.model = model_max
        print('--- Go for Testing ---')
        mse, r2, p_val, CI, logits = self.test_()
        print('Testing MSE: ' + str(mse) +
              ' , Pearson Correlation: ' + str(r2)
              + ' with p-value: ' + str(p_val) +
              ' , Concordance Index: '+str(CI))
        print('--- Training Finished ---')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        # save_dict(path, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        # to support training from multi-gpus data-parallel:

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SS-GNN on PDBbind')
    parser.add_argument('--data', default='data/processed/graphs',
                        help='Path to processed graph directory (default: data/processed/graphs)')
    parser.add_argument('--labels', default='data/processed/labels.pkl',
                        help='Path to labels pickle (default: data/processed/labels.pkl)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent training runs (default: 1). '
                             'Use >1 to replicate the multi-run averaging from the paper.')
    args = parser.parse_args()

    for run in range(args.runs):
        if args.runs > 1:
            print(f'\n=== Run {run + 1}/{args.runs} ===')
        run_suffix = f'_run{run}' if args.runs > 1 else ''
        trainer = Trainer(data_path=args.data, label_path=args.labels, device=args.device)
        trainer.train()
        trainer.save_model(f'saved_models/model{run_suffix}.pt')
