"""Code adapted from https://github.com/gydpku/OCM
"""
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import math
import torch.cuda.amp as amp

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from src.learners.baselines.ocm import OCMLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import OCMResnet, OCMResnet_ImageNet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.utils.losses import WKDLoss


device = get_device()
scaler = amp.GradScaler()

class OCMEMALearner(OCMLearner):
    def __init__(self, args):
        super().__init__(args)

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
        
        # Dist loss
        self.loss_dist = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
        )
        
    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            for j, batch in enumerate(dataloader):
                # print(scaler.get_scale())

                # Stream data
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    self.stream_idx += len(batch_x)

                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    # Augment
                    aug1 = self.rotation(batch_x)
                    aug2 = self.transform_train(aug1)
                    images_pair = torch.cat([aug1, aug2], dim=0)

                    # labels rotations or something
                    rot_sim_labels = torch.cat([batch_y.to(device) + 1000 * i for i in range(self.oop)], dim=0)

                    # Inference
                    feature_map, output_aux = self.model(images_pair, is_simclr=True)
                    simclr = self.normalize(output_aux)
                    feature_map_out = self.normalize(feature_map[:images_pair.shape[0]])
                    
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim1 = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                    temperature=0.07)
                    lo1 = 1 * loss_sim1
                    batch_aug = self.transform_train(batch_x)
                    logits_stu = self.model(batch_aug)
                    logits_stu_raw = self.model(batch_x)
                    
                    loss_ocm = 1*F.cross_entropy(logits_stu, batch_y.long())+1*lo1
                    
                    # adding the distillation part
                    # Augment
                    combined_aug = self.tf_ema(batch_x)
                    
                    # MKD loss
                    logits_stu_raw = self.model(batch_x)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce_mkd = nn.CrossEntropyLoss()(logits_stu, batch_y.long())
                    loss_mkd = loss_ce_mkd + self.params.kd_lambda * loss_dist
                    
                    loss = loss_ocm + loss_mkd

                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)

                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
        else:
            for j, batch in enumerate(dataloader):
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # print(scaler.get_scale())
                    # Stream data
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    self.stream_idx += len(batch_x)
                    
                    # Augment
                    aug1_batch = self.rotation(batch_x)
                    aug2_batch = self.transform_train(aug1_batch)
                    aug1_mem = self.rotation(mem_x)
                    aug2_mem = self.transform_train(aug1_mem)

                    images_pair_batch = torch.cat((aug1_batch, aug2_batch), dim=0)
                    images_pair_mem = torch.cat([aug1_mem, aug2_mem], dim=0)

                    t = torch.cat((images_pair_batch, images_pair_mem),dim=0)
                    feature_map, u = self.model(t, is_simclr=True)
                    pre_u = self.previous_model_ori(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch = self.normalize(feature_map[:images_pair_batch.shape[0]])
                    feature_map_out_mem = self.normalize(feature_map[images_pair_batch.shape[0]:])
                    
                    images_out = u[:images_pair_batch.shape[0]]
                    images_out_r = u[images_pair_batch.shape[0]:]
                    pre_u = self.normalize(pre_u)
                    simclr = self.normalize(images_out)
                    simclr_r = self.normalize(images_out_r)

                    rot_sim_labels = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)

                    num1 = feature_map_out_batch.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id2=torch.randperm(num1)[0]

                    size = simclr.shape[1]

                    sim_matrix = 0.5*torch.matmul(simclr, feature_map_out_batch[:, id1:id1 + size].t())
                    sim_matrix_r = 0.5*torch.matmul(simclr_r,
                                                    feature_map_out_mem[:, id2:id2 + size].t())


                    sim_matrix += 0.5 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r += 0.5 * self.get_similarity_matrix(simclr_r)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):    
                        sim_matrix_r_pre = torch.matmul(simclr_r[:aug1_mem.shape[0]],pre_u.t())

                    loss_sim_r = self.Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
                    loss_sim_pre = self.Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                    X_aug = self.transform_train(mem_x)
                    
                    logits = self.model(X_aug)
                    y_label_pre = self.previous_model_ori(X_aug)
                    loss_pre = 1 * F.mse_loss(y_label_pre[:, self.old_classes.long()], logits[:, self.old_classes.long()])
                    loss_ce = F.cross_entropy(logits, mem_y.long())
                    
                    loss_mi = lo1
                    
                    # adding the distillation part
                    # Augment
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    combined_aug = self.tf_ema(combined_x)

                    # MKD loss
                    logits_stu_raw = self.model(combined_aug)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce_mkd = nn.CrossEntropyLoss()(logits_stu, combined_y.long())
                    loss_mkd = loss_ce_mkd + self.params.kd_lambda * loss_dist
                    
                    loss = loss_ce + loss_mkd + loss_pre + loss_mi
                    
                    
                # Loss
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()
                
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)
                
                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")

                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == self.params.n_tasks - 1:
                        self.tsne(no_drift=True)
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )

        self.previous_model_ori = deepcopy(self.model)
        self.old_classes = torch.cat([self.old_classes, present]).unique()
    
    def train_blurry(self, dataloader, **kwargs):
        self.model.train()
        task_id = -1
        it_since_task_change = self.lag_task_change

        # Stream data
        for j, batch in enumerate(dataloader):
            # update classes seen
            new_class = batch[1].unique().long().to(device)
            curr = torch.cat([self.classes_seen_so_far, new_class]).unique()

            it_since_task_change += 1
            # infer task id
            if len(curr) > len(self.classes_seen_so_far):
                self.old_classes = self.classes_seen_so_far
                self.classes_seen_so_far = curr
                if it_since_task_change >= self.lag_task_change:
                    task_id +=1
                    it_since_task_change = 0
                    if task_id > 0:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            self.previous_model_ori = deepcopy(self.model)

            if task_id == 0:
                # print(scaler.get_scale())

                # Stream data
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    self.stream_idx += len(batch_x)

                    # Augment
                    aug1 = self.rotation(batch_x)
                    aug2 = self.transform_train(aug1)
                    images_pair = torch.cat([aug1, aug2], dim=0)

                    # labels rotations or something
                    rot_sim_labels = torch.cat([batch_y.to(device) + 1000 * i for i in range(self.oop)], dim=0)

                    # Inference
                    feature_map, output_aux = self.model(images_pair, is_simclr=True)
                    simclr = self.normalize(output_aux)
                    feature_map_out = self.normalize(feature_map[:images_pair.shape[0]])
                    
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())

                    sim_matrix += 1 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)

                    loss_sim1 = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                    temperature=0.07)
                    lo1 = 1 * loss_sim1
                    batch_aug = self.transform_train(batch_x)
                    logits_stu = self.model(batch_aug)
                    logits_stu_raw = self.model(batch_x)
                    
                    loss_ocm = 1*F.cross_entropy(logits_stu, batch_y.long())+1*lo1
                    
                    # adding the distillation part
                    # Augment
                    combined_aug = self.tf_ema(batch_x)
                    
                    # MKD loss
                    logits_stu_raw = self.model(batch_x)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce_mkd = nn.CrossEntropyLoss()(logits_stu, batch_y.long())
                    loss_mkd = loss_ce_mkd + self.params.kd_lambda * loss_dist
                    
                    loss = loss_ocm + loss_mkd

                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()

                self.loss = loss.item()
                print(f"Phase : task{task_id}   Loss {self.loss:.3f}  batch {j}", end="\r")
                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())
                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : task{task_id}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
            else:
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # print(scaler.get_scale())
                    # Stream data
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)

                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    self.stream_idx += len(batch_x)
                    
                    # Augment
                    aug1_batch = self.rotation(batch_x)
                    aug2_batch = self.transform_train(aug1_batch)
                    aug1_mem = self.rotation(mem_x)
                    aug2_mem = self.transform_train(aug1_mem)

                    images_pair_batch = torch.cat((aug1_batch, aug2_batch), dim=0)
                    images_pair_mem = torch.cat([aug1_mem, aug2_mem], dim=0)

                    t = torch.cat((images_pair_batch, images_pair_mem),dim=0)
                    feature_map, u = self.model(t, is_simclr=True)
                    pre_u = self.previous_model_ori(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch = self.normalize(feature_map[:images_pair_batch.shape[0]])
                    feature_map_out_mem = self.normalize(feature_map[images_pair_batch.shape[0]:])
                    
                    images_out = u[:images_pair_batch.shape[0]]
                    images_out_r = u[images_pair_batch.shape[0]:]
                    pre_u = self.normalize(pre_u)
                    simclr = self.normalize(images_out)
                    simclr_r = self.normalize(images_out_r)

                    rot_sim_labels = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)

                    num1 = feature_map_out_batch.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id2=torch.randperm(num1)[0]

                    size = simclr.shape[1]

                    sim_matrix = 0.5*torch.matmul(simclr, feature_map_out_batch[:, id1:id1 + size].t())
                    sim_matrix_r = 0.5*torch.matmul(simclr_r,
                                                    feature_map_out_mem[:, id2:id2 + size].t())


                    sim_matrix += 0.5 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r += 0.5 * self.get_similarity_matrix(simclr_r)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):    
                        sim_matrix_r_pre = torch.matmul(simclr_r[:aug1_mem.shape[0]],pre_u.t())

                    loss_sim_r = self.Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
                    loss_sim_pre = self.Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                    X_aug = self.transform_train(mem_x)
                    
                    logits = self.model(X_aug)
                    y_label_pre = self.previous_model_ori(X_aug)
                    loss_pre = 1 * F.mse_loss(y_label_pre[:, self.old_classes.long()], logits[:, self.old_classes.long()])
                    loss_ce = F.cross_entropy(logits, mem_y.long())
                    
                    loss_mi = lo1
                    
                    # adding the distillation part
                    # Augment
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    combined_aug = self.tf_ema(combined_x)

                    # MKD loss
                    logits_stu_raw = self.model(combined_aug)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce_mkd = nn.CrossEntropyLoss()(logits_stu, combined_y.long())
                    loss_mkd = loss_ce_mkd + self.params.kd_lambda * loss_dist
                    
                    loss = loss_ce + loss_mkd + loss_pre + loss_mi
                    
                # Loss
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()
                
                self.loss = loss.item()
                
                print(f"Phase : task{task_id}   Loss {self.loss:.3f}  batch {j}", end="\r")

                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == self.params.n_tasks - 1:
                        self.tsne(no_drift=True)
                    lg.info(
                        f"Phase : task{task_id}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
    
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

        i = 0
        if not self.params.drop_fc:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    logits = self.model_agg(self.transform_test(inputs))
                        
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
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features, _ = self.model_agg(self.transform_test(inputs), is_simclr=True)
                    
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