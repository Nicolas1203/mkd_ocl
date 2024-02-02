'''
Code adapted from https://github.com/FelixHuiweiLin/PCR
'''
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baselines.er import ERLearner
from src.utils.losses import PCRLoss
from src.buffers.reservoir import Reservoir
from src.models.pcr_resnet import PCRResNet18, PCRResNet18_ImageNet
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class PCRLearner(ERLearner):
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
        self.iter = 0
    
    def load_criterion(self):
        return self._criterion
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            return PCRResNet18(
                dim_in=self.params.dim_in,
                n_classes=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return PCRResNet18_ImageNet(
                dim_in=self.params.dim_in,
                n_classes=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model = self.model.train()

        for j, batch_data in enumerate(dataloader):
            # batch update
            batch_x, batch_y = batch_data[0], batch_data[1]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

            self.stream_idx += len(batch_x)

            batch_x_aug = self.transform_train(batch_x).to(device)
            batch_x_combine = torch.cat((batch_x, batch_x_aug))
            batch_y_combine = torch.cat((batch_y, batch_y))
            for i in range(self.params.mem_iters):
                logits, feas= self.model.pcrForward(batch_x_combine)
                novel_loss = 0*self.criterion(logits, batch_y_combine)
                self.optim.zero_grad()

                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                if mem_x.size(0) > 0:
                    mem_x_aug = self.transform_train(mem_x).to(device)
                    mem_x_combine = torch.cat([mem_x, mem_x_aug])
                    mem_y_combine = torch.cat([mem_y, mem_y])


                    mem_logits, mem_fea= self.model.pcrForward(mem_x_combine)

                    combined_feas = torch.cat([mem_fea, feas])
                    combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                    combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels]

                    combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                    combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                    combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                        combined_feas_aug)
                    combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                    cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                combined_feas_aug_normalized.unsqueeze(1)],
                                                dim=1)
                    PSC = PCRLoss(temperature=0.09, contrast_mode='proxy')
                    novel_loss += PSC(features=cos_features, labels=combined_labels).mean()

                novel_loss.backward()
                self.optim.step()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {novel_loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        
    def _criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        return ce(logits, labels)