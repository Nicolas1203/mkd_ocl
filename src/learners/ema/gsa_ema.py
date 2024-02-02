"""Code adapted from https://github.com/gydpku/GSA
Fair warning : the original code is one of the worst I've seen.
Sensitive developpers are advised to not click on the above link.
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

from src.learners.baselines.er import ERLearner
from src.learners.baselines.ocm import OCMLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import GSAResnet, GSAResnet_ImageNet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.utils.losses import WKDLoss

device = get_device()
scaler = amp.GradScaler()

class GSAEMALearner(OCMLearner):
    def __init__(self, args):
        super().__init__(args)
        self.negative_logits_SUM = None
        self.positive_logits_SUM = None
        self.classes_per_task = self.params.n_classes // self.params.n_tasks
        # I know the variable naming is terrible. Please dont judge me it all comes from the authors terrible code
        # One day I will make it look better but its the best I can do rn
        self.Category_sum = None
        self.class_holder = []
        self.tf_gsa = nn.Sequential(
                        RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.)),
                        RandomGrayscale(p=0.2)
                    ).to(device)
        self.flip_num=2
        
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
        
        # Dist loss
        self.loss_dist = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
        )
        self.update_ema(init=True)
    
    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.learning_rate,
            )
        return optimizer
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'imagenet100':
            model = nn.DataParallel(GSAResnet_ImageNet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ))
            return model.to(device)
        else:
            model = nn.DataParallel(GSAResnet(
                    head='mlp',
                    dim_in=self.params.dim_in,
                    dim_int=self.params.dim_int,
                    proj_dim=self.params.proj_dim,
                    n_classes=self.params.n_classes
                ))
            return model.to(device)

    def load_criterion(self):
        return F.cross_entropy
    
    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        new_class_holder = []
        
        if task_id > 0:
            self.Category_sum = torch.cat((self.Category_sum, torch.zeros(self.classes_per_task)))
            self.negative_logits_SUM = torch.cat((self.negative_logits_SUM, torch.zeros(self.classes_per_task).to(device)))
            self.positive_logits_SUM = torch.cat((self.positive_logits_SUM, torch.zeros(self.classes_per_task).to(device)))
            
        negative_logits_sum=None
        positive_logits_sum=None
        sum_num=0
        category_sum = None  
         
        self.model.train()
        for j, batch in enumerate(dataloader):
            # Stream data
            self.optim.zero_grad()
            
            x, y = batch[0].to(device), batch[1].to(device)

            # re-order to adapt GSA code more easily
            y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y]).to(device)
            
            self.stream_idx += len(x)
            
            if not self.buffer.is_empty():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    Y = deepcopy(y)
                    for c in range(len(Y)):
                        if Y[c] not in self.class_holder:
                            self.class_holder.append(Y[c].detach())
                            new_class_holder.append(Y[c].detach())
                    
                    ori_x = x.detach()
                    ori_y = y.detach()
                    x = x.requires_grad_()
                    
                    curr_labels = self.params.labels_order[task_id*self.classes_per_task:(task_id+1)*self.classes_per_task]
                    
                    cur_x, cur_y = self.buffer.only_retrieve(n_imgs=22, desired_labels=curr_labels)
                    cur_x = cur_x.to(device)
                    cur_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in cur_y]).to(device) if len(cur_y) > 0 else cur_y.to(device)
                    
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))
                    
                    # transform
                    x = self.transform_train(x)
                    
                    pred_y = self.model(x)[:, :(task_id+1)*self.classes_per_task]  # Inference 1
                    
                    if task_id>0:
                        pred_y_new = pred_y[:, -self.classes_per_task:]
                    else:
                        pred_y_new=pred_y

                    y_new = y - self.classes_per_task*task_id
                    rate = len(new_class_holder)/len(self.class_holder)

                    mem_x, mem_y = self.buffer.except_retrieve(int(self.params.mem_batch_size*(1-rate)), undesired_labels=curr_labels)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]) if len(mem_y) > 0 else mem_y
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    
                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    mem_x = torch.cat((mem_x[:int(self.params.mem_batch_size*(1-rate))].to(device),index_x[:int(self.params.mem_batch_size*rate)].to(device)),dim=0)
                    mem_y = torch.cat((mem_y[:int(self.params.mem_batch_size*(1-rate))].to(device),index_y[:int(self.params.mem_batch_size*rate)].to(device)))

                    index = torch.randperm(mem_y.size()[0])
                    mem_x = mem_x[index][:]
                    mem_y = mem_y[index][:]

                    mem_y = mem_y.reshape(-1).long()
                    
                    mem_x = mem_x.requires_grad_()
                    
                    mem_x = self.transform_train(mem_x)
                    
                    y_pred = self.model(mem_x)[:, :(task_id+1)*self.classes_per_task]  # Inference 2

                    y_pred_new = y_pred

                    exp_new = torch.exp(y_pred_new)
                    exp_new = exp_new
                    exp_new_sum = torch.sum(exp_new, dim=1)
                    logits_new = (exp_new / exp_new_sum.unsqueeze(1))
                    category_matrix_new = torch.zeros(logits_new.shape)
                    for i_v in range(int(logits_new.shape[0])):
                        category_matrix_new[i_v][mem_y[i_v]] = 1
                    positive_prob = torch.zeros(logits_new.shape)
                    false_prob = deepcopy(logits_new.detach())
                    for i_t in range(int(logits_new.shape[0])):
                        false_prob[i_t][mem_y[i_t]] = 0
                        positive_prob[i_t][mem_y[i_t]] = logits_new[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum is None:
                        negative_logits_sum = torch.sum(false_prob, dim=0)
                        positive_logits_sum = torch.sum(positive_prob, dim=0)
                        if task_id == 0:
                            self.Category_sum = torch.sum(category_matrix_new, dim=0)
                        else:
                            self.Category_sum += torch.sum(category_matrix_new, dim=0)

                        category_sum = torch.sum(category_matrix_new, dim=0)
                    else:
                        self.Category_sum += torch.sum(category_matrix_new, dim=0)
                        negative_logits_sum += torch.sum(false_prob, dim=0)
                        positive_logits_sum += torch.sum(positive_prob, dim=0)
                        category_sum += torch.sum(category_matrix_new, dim=0)
                    if self.negative_logits_SUM is None:
                        self.negative_logits_SUM = torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM = torch.sum(positive_prob, dim=0).to(device)
                    else:
                        self.negative_logits_SUM += torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM += torch.sum(positive_prob, dim=0).to(device)

                    sum_num += int(logits_new.shape[0])
                    
                    if j < 5:
                        ANT = torch.ones(len(self.class_holder))
                    else:
                        ANT = (self.Category_sum.to(device) - self.positive_logits_SUM).to(device)/self.negative_logits_SUM.to(device)

                    ttt = torch.zeros(logits_new.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT):
                            ttt[qqq][mem_y[qqq]] = 1
                        else:
                            ttt[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT[mem_y[qqq]])))

                    loss_n=-torch.sum(torch.log(logits_new)*ttt.to(device))/mem_y.shape[0]
                    loss =2* loss_n + 1 * F.cross_entropy(pred_y_new, y_new.long())
                    
                    # EMA part
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x = mem_x.to(device)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]).to(device)
                    batch_x = batch[0].to(device)
                    batch_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in batch[1]]).to(device)
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    
                    # Augment
                    combined_aug = self.tf_ema(combined_x)
                    
                    # MKD loss
                    logits_stu_raw = self.model(combined_x)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce = self.criterion(logits_stu, combined_y.long())

                    loss += loss_dist * self.params.kd_lambda + loss_ce
                    
                    
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()       
                
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)             

                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")

                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == self.params.n_tasks - 1:
                        self.tsne()
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
            # Update buffer
            self.buffer.update(imgs=batch[0].to(device).detach(), labels=batch[1].to(device).detach())
    
    def train_blurry(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', None)
        new_class_holder = []
        
        self.model.train()
        task_id = -1
        it_since_task_change = self.lag_task_change

        # Stream data
        for j, batch in enumerate(dataloader):
            x, y = batch[0].to(device), batch[1].to(device)

            # re-order to adapt GSA code more easily
            y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y]).to(device)
            
            # update classes seen
            new_class = y.unique().long().to(device)
            curr = torch.cat([self.classes_seen_so_far, new_class]).unique()
            
            if len(self.classes_seen_so_far) > 0:
                diff_labs = curr.max() - self.classes_seen_so_far.max()
            else:
                diff_labs = 0
                
            if diff_labs > 0:
                self.Category_sum = torch.cat((self.Category_sum, torch.zeros(diff_labs)))
                self.negative_logits_SUM = torch.cat((self.negative_logits_SUM, torch.zeros(diff_labs).to(device)))
                self.positive_logits_SUM = torch.cat((self.positive_logits_SUM, torch.zeros(diff_labs).to(device)))
            
            it_since_task_change += 1
            # infer task id
            if len(curr) > len(self.classes_seen_so_far):
                self.old_classes = self.classes_seen_so_far
                self.classes_seen_so_far = curr
                if it_since_task_change >= self.lag_task_change:
                    task_id +=1
                    it_since_task_change = 0
                    
                    negative_logits_sum=None
                    positive_logits_sum=None
                    sum_num=0
                    category_sum = None  
            
            self.model.train()
            # Stream data
            self.optim.zero_grad()
            
            self.stream_idx += len(x)
            
            if not self.buffer.is_empty():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    Y = deepcopy(y)
                    for c in range(len(Y)):
                        if Y[c] not in self.class_holder:
                            self.class_holder.append(Y[c].detach())
                            new_class_holder.append(Y[c].detach())
                    
                    ori_x = x.detach()
                    ori_y = y.detach()
                    x = x.requires_grad_()
                    
                    # Undo labels re-ordering. Labels stored in memory are not ordered.
                    curr_labels = [self.params.labels_order[i] for i in new_class.cpu().tolist()]

                    cur_x, cur_y = self.buffer.only_retrieve(n_imgs=22, desired_labels=curr_labels)
                    cur_x = cur_x.to(device)
                    
                    # Re-order labels coming from memory
                    cur_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in cur_y]).to(device) if len(cur_y) > 0 else cur_y.to(device)
                    
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))
                    
                    # transform
                    x = self.transform_train(x)
                    
                    # pred_y = self.model(x)[:, :(task_id+1)*self.classes_per_task]  # Inference 1
                    max_label = self.classes_seen_so_far.max().item() + 1
                    pred_y = self.model(x)[:, :max_label]  # Inference 1
                    
                    if task_id>0:
                        pred_y_new = pred_y[:, new_class.min():]
                    else:
                        pred_y_new=pred_y

                    # y_new = y - self.classes_per_task*task_id
                    y_new = y - new_class.min()
                    rate = len(new_class_holder)/len(self.class_holder)

                    mem_x, mem_y = self.buffer.except_retrieve(int(self.params.mem_batch_size*(1-rate)), undesired_labels=curr_labels)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]) if len(mem_y) > 0 else mem_y
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    
                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    mem_x = torch.cat((mem_x[:int(self.params.mem_batch_size*(1-rate))].to(device),index_x[:int(self.params.mem_batch_size*rate)].to(device)),dim=0)
                    mem_y = torch.cat((mem_y[:int(self.params.mem_batch_size*(1-rate))].to(device),index_y[:int(self.params.mem_batch_size*rate)].to(device)))

                    index = torch.randperm(mem_y.size()[0])
                    mem_x = mem_x[index][:]
                    mem_y = mem_y[index][:]

                    mem_y = mem_y.reshape(-1).long()
                    
                    mem_x = mem_x.requires_grad_()
                    
                    mem_x = self.transform_train(mem_x)
                    
                    y_pred = self.model(mem_x)[:, :max_label]  # Inference 2

                    y_pred_new = y_pred

                    exp_new = torch.exp(y_pred_new)
                    exp_new_sum = torch.sum(exp_new, dim=1)
                    logits_new = (exp_new / exp_new_sum.unsqueeze(1))
                    category_matrix_new = torch.zeros(logits_new.shape)
                    for i_v in range(int(logits_new.shape[0])):
                        category_matrix_new[i_v][mem_y[i_v]] = 1
                    positive_prob = torch.zeros(logits_new.shape)
                    false_prob = deepcopy(logits_new.detach())
                    for i_t in range(int(logits_new.shape[0])):
                        false_prob[i_t][mem_y[i_t]] = 0
                        positive_prob[i_t][mem_y[i_t]] = logits_new[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum is None:
                        negative_logits_sum = torch.sum(false_prob, dim=0)
                        positive_logits_sum = torch.sum(positive_prob, dim=0)
                        if task_id == 0:
                            self.Category_sum = torch.sum(category_matrix_new, dim=0)
                        else:
                            self.Category_sum += torch.sum(category_matrix_new, dim=0)

                        category_sum = torch.sum(category_matrix_new, dim=0)
                    else:
                        self.Category_sum += torch.sum(category_matrix_new, dim=0)
                        if negative_logits_sum.shape[0] != false_prob.shape[-1]:
                            negative_logits_sum = torch.sum(false_prob, dim=0)
                            positive_logits_sum = torch.sum(positive_prob, dim=0)
                            category_sum = torch.sum(category_matrix_new, dim=0)
                        else:
                            negative_logits_sum += torch.sum(false_prob, dim=0)
                            positive_logits_sum += torch.sum(positive_prob, dim=0)
                            category_sum += torch.sum(category_matrix_new, dim=0)
                    if self.negative_logits_SUM is None:
                        self.negative_logits_SUM = torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM = torch.sum(positive_prob, dim=0).to(device)
                    else:
                        self.negative_logits_SUM += torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM += torch.sum(positive_prob, dim=0).to(device)

                    sum_num += int(logits_new.shape[0])
                    
                    if j < 5:
                        ANT = torch.ones(len(self.class_holder))
                    else:
                        ANT = (self.Category_sum.to(device) - self.positive_logits_SUM).to(device)/self.negative_logits_SUM.to(device)

                    ttt = torch.zeros(logits_new.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT):
                            ttt[qqq][mem_y[qqq]] = 1
                        else:
                            ttt[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT[mem_y[qqq]])))

                    loss_n=-torch.sum(torch.log(logits_new)*ttt.to(device))/mem_y.shape[0]
                    loss =2* loss_n + 1 * F.cross_entropy(pred_y_new, y_new.long())
                    
                    # EMA part
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x = mem_x.to(device)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]).to(device)
                    batch_x = batch[0].to(device)
                    batch_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in batch[1]]).to(device)
                    combined_x = torch.cat([mem_x, batch_x]).to(device)
                    combined_y = torch.cat([mem_y, batch_y]).to(device)
                    
                    # Augment
                    combined_aug = self.tf_ema(combined_x)
                    
                    # MKD loss
                    logits_stu_raw = self.model(combined_x)
                    logits_stu = self.model(combined_aug)
                    loss_dist = 0
                    for teacher in self.ema_models.values():
                        logits_tea = teacher(combined_aug)
                        loss_dist += (self.loss_dist(logits_tea.detach(), logits_stu) + self.loss_dist(logits_tea.detach(), logits_stu_raw))/2
                    loss_dist = loss_dist / len(self.ema_models)
                    loss_ce = self.criterion(logits_stu, combined_y.long())

                    loss += loss_dist * self.params.kd_lambda + loss_ce
                    
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                self.update_ema()                    

                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)
                
                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")

                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == self.params.n_tasks - 1:
                        self.tsne()
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
            # Update buffer
            self.buffer.update(imgs=batch[0].to(device).detach(), labels=batch[1].to(device).detach())
                        
    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        self.init_agg_model()
        i = 0
        if not self.params.drop_fc:
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]

                inputs = inputs.to(device)
                logits = self.model_agg(self.transform_test(inputs))
                preds = logits.argmax(dim=1)
                # transform back to get correct label order
                preds = torch.tensor([self.params.labels_order[i] for i in preds])

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
            return all_feat, all_labels
        else:
            i = 0
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
                _, mem_representations_b = self.model_agg(mem_imgs_b, is_simclr=True)
            else:
                mem_representations_b, _ = self.model_agg(mem_imgs_b, is_simclr=True)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels
    