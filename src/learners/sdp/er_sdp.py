"""
Code adapted from https://github.com/yonseivnl/sdp
"""
import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from copy import deepcopy
from easydict import EasyDict as edict

from src.learners.ce import CELearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18
from src.models.resnet_sdp import ResNet
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device
from src.utils.losses import WKDLoss

device = get_device()

class ER_SDPLearner(CELearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.ema_model1 = deepcopy(self.model)
        self.ema_model2 = deepcopy(self.model)
        self.sdp_model = deepcopy(self.model)
        
        self.mu = self.params.sdp_mu
        self.c2 = self.params.sdp_c2
        
        self.alpha = (1 + np.sqrt(2*self.c2+(2/self.mu)-1))/(self.mu*(1-self.c2)-1)
        self.beta = (1 - np.sqrt(2*self.c2+(2/self.mu)-1))/(self.mu*(1-self.c2)-1)

        print("(alpha, beta)")
        print(self.alpha, self.beta)
        
        self.update_ema(init=True)
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
    
        self.cls_pred_mean = torch.zeros(1).to(self.device)
        self.cls_pred = {}
        self.cls_pred_length = 100

    def load_model(self, **kwargs):
        opt = edict(
            {
                "depth": 18,
                "num_classes": self.params.n_classes,
                "in_channels": 3,
                "bn": True,
                "normtype": "BatchNorm",
                "activetype": "ReLU",
                "pooltype": "MaxPool2d",
                "preact": False,
                "affine_bn": True,
                "bn_eps": 1e-6,
                "compression": 0.5,
            }
        )
        model = ResNet(opt)
        model.to(self.device)
        return model
    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    
                    # re-order to adapt SDP code more easily
                    combined_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in combined_y.long()]).to(device)
                    
                    present = combined_y.unique().to(device)
                    self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique()
                    
                    # Augment
                    combined_aug = self.transform_train(combined_x)
                    
                    # SDP part
                    # Inference
                    logits, feature = self.model(combined_aug, get_feature=True)
                    logits = logits[:, :(self.classes_seen_so_far.max()+1)]
                    
                    cls_loss = self.criterion(logits, combined_y.long())
                    self.sdp_model.zero_grad()
                    with torch.no_grad():
                        logit2, feature2 = self.sdp_model(combined_aug, get_feature=True)
                    distill_loss = ((feature - feature2.detach()) ** 2).sum(dim=1)
                    self.update_cls_pred(xs=combined_aug, ys=combined_y.long())
                    sample_weight = self.cls_pred_mean
                    grad = self.get_grad(logits.detach(), combined_y.long(), self.model.fc.weight)
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    loss = ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
                    
                    # Loss
                    self.loss = loss.item()
                    if not self.params.no_wandb:
                        wandb.log({
                            "loss": self.loss
                        })
                    print(f"Loss : {loss.item():.4f}   cls_loss:{cls_loss.mean().item():.4f}  dist_loss:{distill_loss.mean().item():.4f}")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.update_ema()

            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}   cls_loss:{cls_loss.mean().item():.4f}  dist_loss:{distill_loss.mean().item():.4f}  time : {time.time() - self.start:.4f}s"
                )
                
    
    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
    
    def evaluate(self, dataloaders, task_id):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            self.model.eval()
            accs = []

            all_preds = []
            all_targets = []
            for j in range(task_id + 1):
                test_preds, test_targets = self.encode(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)
                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    all_preds = np.concatenate([all_preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        f"acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                all_preds = [self.params.labels_order.index(int(i)) for i in all_preds]
                n_im_pt = self.params.n_classes // self.params.n_tasks
                all_targets_b = [0 if i < (task_id)*n_im_pt else 1 for i in all_targets]
                all_preds_b = [0 if i < (task_id)*n_im_pt else 1 for i in all_preds]
                cm = confusion_matrix(all_targets, all_preds)
                mcc = matthews_corrcoef(all_targets, all_preds)
                mcc_b = matthews_corrcoef(all_targets_b, all_preds_b)
                cm_log = np.log(1 + cm)
                fig = plt.matshow(cm_log)
                wandb.log({
                        "cm_raw": cm,
                        "cm": fig,
                        "mcc": mcc,
                        "mcc_b": mcc_b,
                        "task_id": task_id
                    })
                
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

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        
    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                logits = self.model(self.transform_test(inputs))
                preds = logits.argmax(dim=1)
                # transform back to get correct label order
                preds = torch.tensor([self.params.labels_order[i] for i in preds])
                
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels
    
    @torch.no_grad()
    def update_cls_pred(self, xs, ys):
        for x, y in zip(xs, ys):
            self.model.eval()
            logit = self.model(x.unsqueeze(0))
            prob = F.softmax(logit, dim=1)
            if y.item() not in self.cls_pred.keys():
                self.cls_pred[y.item()] = []
            self.cls_pred[y.item()].append(prob[0, y].item())
            if len(self.cls_pred[y.item()]) > self.cls_pred_length:
                del self.cls_pred[y.item()][0]
            self.cls_pred_mean = np.clip(np.mean([np.mean(cls_pred) for cls_pred in self.cls_pred.values()]) - 1/(self.classes_seen_so_far.max().cpu()+1), 0, 1) * (self.classes_seen_so_far.max()+1)/(self.classes_seen_so_far.max()+ 2)
        self.model.train()
        
    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), (self.classes_seen_so_far.max()+1))
        return torch.matmul((prob - oh_label), weight[:(self.classes_seen_so_far.max()+1), :])
    
    @torch.no_grad()
    def update_ema(self, init=False):
        """
        Update the Exponential Moving Average (EMA) of both PT models
        """
        for param, ema_param1, ema_param2, sdp_param in zip(self.model.parameters(), self.ema_model1.parameters(), self.ema_model2.parameters(), self.sdp_model.parameters()):
            p = deepcopy(param.data.detach())
            if init:
                ema_param1.data.mul_(0).add_(p * self.alpha)
                ema_param2.data.mul_(0).add_(p * self.beta)
                sdp_param.data.mul_(0).add_((self.alpha / (self.alpha - self.beta))*ema_param2.data.detach() - (self.beta / (self.alpha - self.beta))*ema_param1.data.detach())
            else:
                ema_param1.data.mul_(1 - self.alpha).add_(p * self.alpha)
                ema_param2.data.mul_(1 - self.beta).add_(p * self.beta)
                sdp_param.data.mul_(0).add_((self.alpha / (self.alpha - self.beta))*ema_param2.data.detach() - (self.beta / (self.alpha - self.beta))*ema_param1.data.detach())
                
