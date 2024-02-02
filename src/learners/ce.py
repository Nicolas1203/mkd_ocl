import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.models.resnet import SupConResNet
from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils.metrics import forgetting_line   


class CELearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []

    def load_model(self):
        model = SupConResNet(
            head='mlp',
            dim_in=self.params.dim_in,
            proj_dim=self.params.n_classes,
            input_channels=self.params.nb_channels,
            nf=self.params.nf
        )
        # model = nn.DataParallel(model)
        model.to(self.device)
        return model
        
    def load_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, dataloader, epoch=1, **kwargs):
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            self.stream_idx += len(batch_x)
            
            batch_aug = self.transform_train(batch_x)

            # Inference
            _, logits = self.model(batch_aug)  # (batch_size, projection_dim)
            # preds = nn.Softmax(dim=1)(logits)

            # Loss
            loss = self.criterion(logits, batch_y.long())
            self.loss = loss.item() 
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Epoch : {epoch}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def evaluate(self, dataloaders, task_id):
        self.model.eval()
        accs = []
        fgts = []

        for j in range(task_id + 1):
            test_preds, test_targets = self.encode(dataloaders[f"test{j}"])
            acc = accuracy_score(test_preds, test_targets)
            accs.append(acc)
        for _ in range(self.params.n_tasks - task_id - 1):
            accs.append(np.nan)
        self.results.append(accs)
        
        line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
        line = line[0].to_numpy().tolist()
        self.results_forgetting.append(line)

        self.print_results(task_id)

        return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1])
    
    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")

    def save_results(self):
        results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")
        print(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        df_avg = pd.DataFrame()
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        # Loop over classifiers results
        # Each classifier has a value for every task. NaN if future task
        df_clf = pd.DataFrame(self.results, columns=cols)
        # Average accuracy over all tasks with not NaN value
        df_avg = df_clf.mean(axis=1)
        df_clf.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
        
        df_avg = pd.DataFrame()
        print(f"Saving forgetting results in : {results_dir}")
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        df_clf = pd.DataFrame(self.results_forgetting, columns=cols)
        df_avg = df_clf.mean(axis=1)
        df_clf.to_csv(os.path.join(results_dir, 'forgetting.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

        self.save_parameters()

    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
            if not ((epoch + 1) % 5):
                self.save(model_name=f"CE_{epoch}.pth")
        
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]

    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                _, logits = self.model(self.transform_test(inputs))
                preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
                i += 1
        return all_feat, all_labels

    def save_results_offline(self):
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)

        print(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        pd.DataFrame(self.results).to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

        self.save_parameters()

    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)