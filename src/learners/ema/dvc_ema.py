"""adapted from https://github.com/YananGu/DVC
"""
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from torch.utils.data import DataLoader

from src.learners.baselines.dvc import DVCLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.utils.augment import MixUpAugment, CutMixAugment
from src.models.resnet import DVCNet, DVCNet_ImageNet
from src.utils.utils import get_device
from src.utils.losses import WKDLoss

device = get_device()

class DVCEMALearner(DVCLearner):
    def __init__(self, args):
        super().__init__(args)
         # Dist loss
        self.loss_dist = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
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
        
        self.tf_ema = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
        ).to(device)
        
        self.update_ema(init=True)
    
    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'Unknown task name')
        task_id = kwargs.get("task_id", None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0].to(device), batch[1].long().to(device)
            batch_x_aug = self.transform_train(batch_x)
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                y = self.model(batch_x, batch_x_aug)
                z, zt, _,_ = y
                ce = self.cross_entropy_loss(z, zt, batch_y, label_smoothing=0)

                agreement_loss, dl = self.agmax_loss(y, batch_y, dl_weight=self.dl_weight)
                loss  = ce + agreement_loss + dl

                # backward
                self.optim.zero_grad()
                loss.backward()

                mem_x, mem_x_aug, mem_y = self.buffer.mgi_retrieve(
                    n_imgs=self.params.mem_batch_size,
                    out_dim=self.params.n_classes,
                    model=self.model
                    )

                if mem_x.size(0) > 0:
                    mem_x, mem_x_aug, mem_y = mem_x.to(device), mem_x_aug.to(device), mem_y.to(device)
                    y = self.model(mem_x, mem_x_aug)
                    z, zt, _,_ = y
                    ce = self.cross_entropy_loss(z, zt, mem_y, label_smoothing=0)
                    agreement_loss, dl = self.agmax_loss(y, mem_y, dl_weight=self.dl_weight)
                    loss_mem = ce  + agreement_loss + dl

                    loss_mem.backward()
                    
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    
                    # MKD loss
                    # Augment
                    combined_aug = self.tf_ema(combined_x)
                    logits_stu_raw, _, _, _ = self.model(combined_x, combined_x)
                    logits_stu, _, _, _ = self.model(combined_aug, combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea, _, _, _ = teacher(combined_aug, combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce = nn.CrossEntropyLoss()(logits_stu, combined_y.long())
                    
                    loss_mkd = self.params.kd_lambda*loss_dist + loss_ce
                    
                    loss_mkd.backward()
                    self.loss = (loss.mean() + loss_mkd.mean()).item()
                    
                    self.update_ema()                    
                    
                self.optim.step()
                
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)
                    
            # Update buffer
            print(f"Loss {self.loss:.3f} batch {j}", end="\r")
            self.buffer.update(imgs=batch_x, labels=batch_y)
            if (j == (len(dataloader) - 1)) and (j > 0):
                if self.params.tsne and task_id == self.params.n_tasks - 1:
                    self.tsne()
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def encode(self, dataloader, nbatches=-1, **kwargs):
        self.model_agg = deepcopy(self.model)
        # infer with model_agg as average of all the ema models
        with torch.no_grad():
            for teacher in self.ema_models.values():
                for param_agg, teacher_param in zip(self.model_agg.parameters(), teacher.parameters()):
                    param_agg.add_(teacher_param.detach())
            for param_agg in self.model_agg.parameters():
                param_agg.mul_(1/(len(self.ema_models) + 1))
        self.model_agg.eval()
        
        i = 0
        if not self.params.drop_fc:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(device)
                    inputs = self.transform_test(inputs)
                    logits = self.model_agg(inputs, inputs)[0]
                    preds = logits.argmax(dim=1)

                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = preds.cpu().numpy()
                        i += 1
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.hstack([all_feat, preds.cpu().numpy()])
            return all_feat, all_labels
        else:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features = self.model_agg.backbone(self.transform_test(inputs))[0]
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
    
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