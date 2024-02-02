
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import torchvision

from sklearn.metrics import accuracy_score
from copy import deepcopy

from src.learners.baselines.derpp import DERppLearner
from src.buffers.logits_res import LogitsRes
from src.models.resnet import ResNet18
from src.utils.metrics import forgetting_line
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device
from src.utils.losses import WKDLoss


device = get_device()

class DERppEMALearner(DERppLearner):
    def __init__(self, args):
        super().__init__(args)
        # Dist loss
        self.loss_dist = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
        )

        # Ema model
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
        
        self.tf_ema = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
        ).to(device)
        
        self.update_ema(init=True)

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

    def load_criterion(self):
        return F.cross_entropy
    
    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc' or self.params.training_type == "blurry":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
        
    def train_inc(self, dataloader, task_name=None, **kwargs):
        """Adapted from https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
        """
        task_id = kwargs.get("task_id", None)
        for j, batch in enumerate(dataloader):
            self.model = self.model.train()
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                outputs = self.model.logits(self.transform_train(batch_x.to(device)))
                loss = self.criterion(outputs, batch_y.long().to(device))
                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                if mem_x.size(0) > 0:
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    
                    # Loss
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_beta * self.criterion(mem_outputs, mem_y.long().to(device))

                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    # Augment
                    combined_aug = self.tf_ema(combined_x)
                    
                    # MKD loss
                    logits_stu_raw = self.model.logits(combined_x)
                    logits_stu = self.model.logits(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher.logits(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce = nn.CrossEntropyLoss()(logits_stu, combined_y.long())
                    
                    loss_mkd = self.params.kd_lambda*loss_dist + loss_ce
                    
                    loss += loss_mkd
                    
                    self.loss = loss.mean().item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    loss.backward()
                    self.optim.step()
                    
                    self.update_ema()
                    
                    if self.params.measure_drift >=0 and task_id > 0:
                        self.measure_drift(task_id)

            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=outputs.detach())
            if (j == (len(dataloader) - 1)) and (j > 0):
                if self.params.tsne and task_id == self.params.n_tasks - 1:
                    self.tsne()
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def encode(self, dataloader, nbatches=-1, **kwargs):
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
        self.model_agg = deepcopy(self.model)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # infer with model_agg as average of all the ema models
            with torch.no_grad():
                for teacher in self.ema_models.values():
                    for param_agg, teacher_param in zip(self.model_agg.parameters(), teacher.parameters()):
                        param_agg.add_(teacher_param.detach())
                for param_agg in self.model_agg.parameters():
                    param_agg.mul_(1/(len(self.ema_models) + 1))
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

    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
