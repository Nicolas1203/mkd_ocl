
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

from src.learners.sdp.er_sdp import ER_SDPLearner
from src.buffers.logits_res import LogitsRes
from src.models.resnet import ResNet18
from src.utils.metrics import forgetting_line
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device
from src.utils.losses import WKDLoss


device = get_device()

class DERpp_SDPLearner(ER_SDPLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = LogitsRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        
        self.update_ema(init=True)

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
        
    def train(self, dataloader, task_name=None, **kwargs):
        """Adapted from https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
        """
        for j, batch in enumerate(dataloader):
            self.model = self.model.train()
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                outputs = self.model(self.transform_train(batch_x.to(device)))
                loss = self.criterion(outputs, batch_y.long().to(device))
                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                if mem_x.size(0) > 0:
                    mem_outputs = self.model(self.transform_train(mem_x.to(device)))
                    
                    # Loss
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_outputs = self.model(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_beta * self.criterion(mem_outputs, mem_y.long().to(device))

                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    
                    present = combined_y.unique().to(device)
                    self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique().long()
                    
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
                    loss += ((1 - sample_weight) * cls_loss + beta * sample_weight * distill_loss).mean()
                    
                    self.loss = loss.mean().item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    loss.backward()
                    self.optim.step()
                    
                    self.update_ema()

            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=outputs.detach())

            if (j == (len(dataloader) - 1)) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )