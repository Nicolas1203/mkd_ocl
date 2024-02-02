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

device = get_device()
scaler = amp.GradScaler()

class GSALearner(OCMLearner):
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
    
    def RandomFlip(self, x, num=0):
        if not num:
            num=self.flip_num
        x=self.tf_gsa(x)
        X = []

        X.append(x)
        X.append(self.flip_inner(x, 1, 1))

        X.append(self.flip_inner(x, 0, 1))

        X.append(self.flip_inner(x, 1, 0))

        return torch.cat([X[i] for i in range(num)], dim=0)
    
    def flip_inner(self, x, flip1, flip2):
        num = x.shape[0]

        a = x  # .permute(0,1,3,2)
        a = a.view(num, 3, 2, 16, 32)
        a = a.permute(2, 0, 1, 3, 4)
        s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
        s2 = a[1]  # .permute(1,0, 2, 3)
        if flip1:
            s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
        if flip2:
            s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

        s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
        S = s.reshape(num, 3, 32, 32)
        return S
    
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
                    
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                
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
                
            if diff_labs > 0 and self.Category_sum is not None:
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
                    
                    curr_labels = [self.params.labels_order[i] for i in new_class.cpu().tolist()]

                    cur_x, cur_y = self.buffer.only_retrieve(n_imgs=22, desired_labels=curr_labels)
                    cur_x = cur_x.to(device)
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
                    
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()

                self.loss = loss.item()
                print(f"Phase : task{task_id} Loss {self.loss:.3f}  batch {j}", end="\r")

                # Plot to tensorboard
                if (j == (len(dataloader) - 1)) and (j > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
            # Update buffer
            self.buffer.update(imgs=batch[0].to(device).detach(), labels=batch[1].to(device).detach())

    def encode(self, dataloader, nbatches=-1, **kwargs):
        i = 0
        if not self.params.drop_fc:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(device)
                    logits = self.model(self.transform_test(inputs))
                    preds = logits.argmax(dim=1)
                    preds = torch.tensor([self.params.labels_order[i] for i in preds])

                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = preds.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.hstack([all_feat, preds.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
        else:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features, _ = self.model(self.transform_test(inputs), is_simclr=True)
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels