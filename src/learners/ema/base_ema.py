import torch
import time
import torch.nn as nn
import sys
import logging as lg
import pandas as pd
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.utils.losses import WKDLoss
from src.utils.augment import MixUpAugment, CutMixAugment
from src.utils.utils import filter_labels
from src.utils.utils import get_device

device = get_device()

class BaseEMALearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        
        # Dist loss
        self.loss_dist = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
        )

        # Ema model
        self.ema_model1 = deepcopy(self.model)
        self.ema_model2 = deepcopy(self.model)
        self.ema_model3 = deepcopy(self.model)
        self.ema_model4 = deepcopy(self.model)
        
        self.ema_alpha1 = 0.1
        self.ema_alpha2 = 0.05
        self.ema_alpha3 = 0.01
        self.ema_alpha4 = 0.005
        
        self.update_ema(init=True)
        
    def load_criterion(self):
        return SupConLoss(self.params.temperature)
    
    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc' or self.params.training_type == "blurry":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
        
    def train_inc(self, dataloader, **kwargs):
        raise NotImplementedError
        
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

    def update_ema(self, init=False):
        """
        Update the Exponential Moving Average (EMA) of a PyTorch model.

        Args:
        - model: the original model whose weights will be EMA-ed
        - model_ema: the EMA model that will be updated
        - alpha: the EMA decay rate

        Returns:
        - None
        """
        # Iterate through each parameter in the model and EMA model
        # for (name, param), ema_param1, ema_param2 in zip(self.model.named_parameters(), self.ema_model1.parameters(), self.ema_model2.parameters()):
        # for (name, param), ema_param1, ema_param2, ema_param3 in zip(self.model.named_parameters(), self.ema_model1.parameters(), self.ema_model2.parameters(), self.ema_model3.parameters()):
        # for (name, param), ema_param1, ema_param2, ema_param3, ema_param4 in zip(self.model.named_parameters(), self.ema_model1.parameters(), self.ema_model2.parameters(), self.ema_model3.parameters(), self.ema_model4.parameters()):
        for (name, param), ema_param1, ema_param2, ema_param3, ema_param4 in zip(self.model.named_parameters(), self.ema_model1.parameters(), self.ema_model2.parameters(), self.ema_model3.parameters(), self.ema_model4.parameters()):
            # Detach the EMA parameter from the graph to prevent gradient updates
            # ema_param1.detach_()
            # ema_param2.detach_()
            # ema_param3.detach_()
            # ema_param4.detach_()
            
            alpha1 = self.ema_alpha1
            alpha2 = self.ema_alpha2
            alpha3 = self.ema_alpha3
            alpha4 = self.ema_alpha4
            
            # p = deepcopy(param.data.detach())
            p = deepcopy(param.data.detach())
            
            if init:
                ema_param1.data.mul_(0).add_(p * alpha1 / (1 - alpha1 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param2.data.mul_(0).add_(p * alpha2 / (1 - alpha2 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param3.data.mul_(0).add_(p * alpha3 / (1 - alpha3 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param4.data.mul_(0).add_(p * alpha4 / (1 - alpha4 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
            else:
                # ema_param1.data.mul_(1 - alpha).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param1.data.mul_(1 - alpha1).add_(p * alpha1 / (1 - alpha1 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param2.data.mul_(1 - alpha2).add_(p * alpha2 / (1 - alpha2 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param3.data.mul_(1 - alpha3).add_(p * alpha3 / (1 - alpha3 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                ema_param4.data.mul_(1 - alpha4).add_(p * alpha4 / (1 - alpha4 ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
