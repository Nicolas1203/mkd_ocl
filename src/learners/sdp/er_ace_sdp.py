"""Adapted from https://github.com/pclucas14/AML/blob/paper_open_source/methods/er_ace.py
and https://github.com/pclucas14/AML/blob/7c929363d9c687e0aa4539c1ab91c812330d421f/methods/er.py#L10
"""
import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from copy import deepcopy
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from src.learners.sdp.er_sdp import ER_SDPLearner
from src.learners.ema.base_ema import BaseEMALearner
from src.utils.losses import SupConLoss
from src.utils import name_match
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device
from src.utils.losses import WKDLoss

device = get_device()

class ER_ACE_SDPLearner(ER_SDPLearner):
    def __init__(self, args):
        super().__init__(args)
        
    def load_criterion(self):
        return F.cross_entropy

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'Unknown task name')
        self.model = self.model.train()
        present = torch.LongTensor(size=(0,)).to(device)

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1].long()
            self.stream_idx += len(batch_x)
            
            # update classes seen
            present = batch_y.unique().to(device)
            self.classes_seen_so_far = torch.cat([self.classes_seen_so_far, present]).unique()
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                # process stream
                aug_xs = self.transform_train(batch_x.to(device))
                logits = self.model(aug_xs)
                mask = torch.zeros_like(logits).to(device)

                # unmask curent classes
                mask[:, present] = 1
                
                # unmask unseen classes
                unseen = torch.arange(len(logits)).to(device)
                for c in self.classes_seen_so_far:
                    unseen = unseen[unseen != c]
                mask[:, unseen] = 1    

                logits_stream = logits.masked_fill(mask == 0, -1e9)   
                loss = self.criterion(logits_stream, batch_y.to(device))

                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Augment
                    aug_xm = self.transform_train(mem_x).to(device)

                    # Inference
                    logits_mem = self.model(aug_xm)
                    loss += self.criterion(logits_mem, mem_y.to(device))
                
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    
                    # Augment
                    combined_aug = self.transform_train(combined_x)
                    
                    # SDP Loss
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
                
                # Loss
                self.loss = loss.item()
                
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                loss.backward()
                self.optim.step()

                self.update_ema()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
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