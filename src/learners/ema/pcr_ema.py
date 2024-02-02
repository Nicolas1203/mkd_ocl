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

from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baselines.pcr import PCRLearner
from src.utils.losses import PCRLoss, WKDLoss
from src.buffers.reservoir import Reservoir
from src.models.pcr_resnet import PCRResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

device = get_device()

class PCR_EMALearner(PCRLearner):
    def __init__(self, args):
        super().__init__(args)
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
            alpha_kd=self.params.alpha_kd
        )
        
        self.ema_models = {}
        self.ema_alphas = {}
        if self.params.alpha_min is None or self.params.alpha_max is None:
            # Manually set 5 teachers
            self.ema_models[0] = deepcopy(self.model)
            self.ema_models[1] = deepcopy(self.model)
            self.ema_models[2] = deepcopy(self.model)
            self.ema_models[3] = deepcopy(self.model)
            
            self.ema_alphas[0] = self.params.ema_alpha1
            self.ema_alphas[1] = self.params.ema_alpha2
            self.ema_alphas[2] = self.params.ema_alpha3
            self.ema_alphas[3] = self.params.ema_alpha4
        else:
            # Automatically set a specific number of teachers
            for i in range(self.params.n_teacher):
                self.ema_models[i] = deepcopy(self.model)
                self.ema_alphas[i] = 10**(
                    np.log10(self.params.alpha_min) + 
                    (np.log10(self.params.alpha_max) - np.log10(self.params.alpha_min))*i/(max(1,self.params.n_teacher - 1))
                    )
        print(self.ema_alphas)
        self.update_ema(init=True)
        
    def load_criterion(self):
        return self._criterion
    
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
                    
                    
                    # Distillation
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    combined_aug = self.transform_train(combined_x)
                    
                    # logits
                    logits_stu = self.model.pcrForward(combined_aug)[0]
                    logits_stu_raw = self.model.pcrForward(combined_x)[0]
                    
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher.pcrForward(combined_aug)[0]
                        loss_dist += (self.wkdloss(logits_tea.detach(), logits_stu) + self.wkdloss(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    
                    loss_ce = self.criterion(logits_stu, combined_y.long())
                    loss_mkd = self.params.kd_lambda*loss_dist + loss_ce
                    
                    loss = novel_loss + loss_mkd
                    
                    loss.backward()
                self.optim.step()
                self.update_ema()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {novel_loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    
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

    def update_ema(self, init=False):
        """
        Update the Exponential Moving Average (EMA) of the group of pytorch models
        """
        for i, ema_model in enumerate(self.ema_models.values()):
            alpha = self.ema_alphas[i]
            for param, ema_param in zip(self.model.parameters(), ema_model.parameters()):
                p = deepcopy(param.data.detach())
                
                if init:
                    ema_param.data.mul_(0).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                else:
                    ema_param.data.mul_(1 - alpha).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))

    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        self.model_agg = deepcopy(self.model)
        with torch.no_grad():
            for teacher in self.ema_models.values():
                for param_agg, teacher_param in zip(self.model_agg.parameters(), teacher.parameters()):
                    param_agg.add_(teacher_param.detach())
            for param_agg in self.model_agg.parameters():
                param_agg.mul_(1/(len(self.ema_models) + 1))
                
        self.model_agg.eval()
        if not self.params.drop_fc:
            i = 0
                
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)

                    logits = self.model_agg.pcrForward(self.transform_test(inputs))[0]
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