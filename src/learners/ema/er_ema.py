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

class ER_EMALearner(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
            alpha_kd=self.params.alpha_kd
        )

        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        
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
        
        self.previous_model = None
        if self.params.measure_drift >= 0:
            self.drift = []
            self.previous_model = None
            
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
                        logits_stu = self.model.logits(combined_aug)
                        logits_stu_raw = self.model.logits(combined_x)
                        
                        loss_dist = 0
                        for teacher in self.ema_models.values():
                            logits_tea = teacher.logits(combined_aug)
                            if self.params.no_aug:
                                loss_dist += (self.wkdloss(logits_tea.detach(), logits_stu))
                            else:
                                loss_dist += (self.wkdloss(logits_tea.detach(), logits_stu) + self.wkdloss(logits_tea.detach(), logits_stu_raw))/2
                        loss_dist = loss_dist / len(self.ema_models)
                        
                        loss_ce = self.criterion(logits_stu, combined_y.long())
                        loss = self.params.kd_lambda*loss_dist + loss_ce
                            
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.update_ema()
                        if self.params.annealing:
                            self.scheduler.step()
                        self.optim.zero_grad()
                        
                        if self.params.measure_drift >=0 and task_id > 0:
                            self.measure_drift(task_id)
                        
                        if not self.params.no_wandb:
                            wandb.log({
                                "loss_dist": loss_dist.item(),
                                "loss": loss.item()
                            })
                        print(f"Phase: {task_name}  Loss:{loss.item():.3f}  Loss dist:{loss_dist.item():.3f} batch {j}", end="\r")
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
        if self.params.eval_teacher:
            self.model_agg = deepcopy(list(self.ema_models.values())[0])
        else:
            self.model_agg = deepcopy(self.model)
            if not self.params.no_avg:
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