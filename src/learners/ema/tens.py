import torch
import time
import torch.nn as nn
import sys
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import torch.cuda.amp as amp
import random
import wandb
import matplotlib.pyplot as plt

from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

from src.learners.base import BaseLearner
from src.learners.baselines.er import ERLearner
from src.utils.losses import WKDLoss 
from src.models.resnet import ResNet18
from src.utils import name_match
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device, filter_labels
from src.utils.augment import MixupAdaptative, ZetaMixup

device = get_device()

scaler = amp.GradScaler()

LR_MIN = 5e-4
LR_MAX = 5e-2

class TEns(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        
        # Manually set EMA ensemble
        self.ema_model = deepcopy(self.model)
        self.ema_alpha = self.params.alpha_min
        self.n_updates = 0
        
        print(self.ema_alpha)
        self.update_ema(init=True)
        self.params.eval_teacher = True # Temporal Ensemble evaluates with the EMA model
        
    # @profile
    def train(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name", "Unknown")
        task_id = kwargs.get('task_id', None)
        self.model = self.model.train()
        
        for j, batch in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # Stream batch
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                # self.ema_model1.train()
                
                for _ in range(self.params.mem_iters):
                    # Iteration over memory + stream
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    
                    if mem_x.size(0) > 0:
                        combined_x = torch.cat([mem_x, batch_x]).to(device)
                        combined_y = torch.cat([mem_y, batch_y]).to(device)
                        
                        # Augment
                        combined_aug = self.transform_train(combined_x)
                        
                        # logits
                        # Do a forward pass to update Batch Norm parameters of ema model
                        _ = self.ema_model.logits(combined_aug) 
                        
                        logits_stu = self.model.logits(combined_aug)
                        loss_ce = self.criterion(logits_stu, combined_y.long())
                        loss = loss_ce
                            
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.update_ema()
                        self.optim.zero_grad()
                        
                        if not self.params.no_wandb:
                            wandb.log({
                                "loss": loss.item()
                            })
                        print(f"Phase: {task_name}  Loss:{loss.item():.3f} batch {j}", end="\r")
                self.buffer.update(imgs=batch_x, labels=batch_y)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == 4:
                        self.tsne()
                    print(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s",
                        end="\r"
                    )
                    self.save(model_name=f"ckpt_{task_name}.pth")
    
    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
    
    def calculate_ema_weights(self, n, alpha):
        ema_weights = []  # Initialize weight for the first data point as 1
        for i in range(1, n):
            ema_weights.append(alpha * (1 - alpha) ** i)  # Calculate weight for each data point
        # ema_weights.append((1 - alpha) ** n)
        return ema_weights

    def update_ema(self, init=False):
        """
        Update the Exponential Moving Average (EMA) of the group of pytorch models
        """
        self.n_updates += 1
        alpha = self.ema_alpha
        norm_w = np.array(self.calculate_ema_weights(self.n_updates, alpha)).sum()
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            p = deepcopy(param.data.detach())
            if init:
                ema_param.data.mul_(0).add_(p * alpha).mul(1/norm_w)
            else:
                ema_param.data.mul_(1 - alpha).add_(p * alpha).mul(1/norm_w)
                # ema_param.data.mul_(1 - alpha).add_(p * alpha)

    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        self.init_agg_model()
        if not self.params.drop_fc:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)

                    logits = self.model_agg.logits(self.transform_test(inputs))
                    preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_preds = preds.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_preds = np.hstack([all_preds, preds.cpu().numpy()])
                    i += 1
            
            return all_preds, all_labels
        else:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features, _ = self.model_agg(self.transform_test(inputs))
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
    
    def init_agg_model(self):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                self.model_agg = deepcopy(self.ema_model)
                self.model_agg.eval()
    
    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        self.init_agg_model()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            if use_proj:
                _, mem_representations_b = self.model_agg(mem_imgs_b)
            else:
                mem_representations_b, _ = self.model_agg(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels