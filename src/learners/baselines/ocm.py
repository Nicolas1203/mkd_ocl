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
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale

from src.learners.baselines.er import ERLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import OCMResnet, OCMResnet_ImageNet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device

device = get_device()
scaler = amp.GradScaler()

class OCMLearner(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,  
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        # When task id need to be infered
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        self.old_classes = torch.LongTensor(size=(0,)).to(device)
        self.lag_task_change = 100

        self.oop = 16
        
    def rotation(self, x):
        X = self.rot_inner_all(x)#, 1, 0)
        return torch.cat((X,torch.rot90(X,2,(2,3)),torch.rot90(X,1,(2,3)),torch.rot90(X,3,(2,3))),dim=0)

    def rot_inner_all(self, x):
        num=x.shape[0]
        R=x.repeat(4,1,1,1)
        a=x.permute(0,1,3,2)
        a = a.view(num,3, 2, self.params.img_size // 2 , self.params.img_size)
        a = a.permute(2,0, 1, 3, 4)
        s1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2=a[1]#.permute(1,0, 2, 3)
        a= torch.rot90(a, 2, (3, 4))
        s1_1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2_2=a[1]#.permute(1,0, 2, 3)R[3*num:]

        R[num:2*num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[3*num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        return R

    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.99),
            weight_decay=1e-4
            )
        return optimizer
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'imagenet100':
            model = nn.DataParallel(OCMResnet_ImageNet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ))
            return model.to(device)
        else:
            model = nn.DataParallel(OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ))
            return model.to(device)

    def load_criterion(self):
        return SupConLoss(self.params.temperature) 
    
    def normalize(self, x, dim=1, eps=1e-8):
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def get_similarity_matrix(self, outputs, chunk=2, multi_gpu=False):
        '''
            Compute similarity matrix
            - outputs: (B', d) tensor for B' = B * chunk
            - sim_matrix: (B', B') tensor
        '''
        sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')#这里是sim(z(x),z(x'))
        return sim_matrix
    
    def Supervised_NT_xent_n(self, sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        eye = torch.eye(B * chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)
        return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) +  loss1#+1*loss2
    
    def Supervised_NT_xent_uni(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature)# * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

    def Supervised_NT_xent_pre(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels#.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature) #* (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

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
                    y_pred = self.model(self.transform_train(batch_x))

                    loss = 1*F.cross_entropy(y_pred, batch_y.long())+1*lo1

                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
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

                    y_label = self.model(self.transform_train(mem_x))

                    y_label_pre = self.previous_model_ori(self.transform_train(mem_x))
                    loss = 1 * F.cross_entropy(y_label, mem_y) + lo1  + \
                        1 * F.mse_loss(y_label_pre[:, self.old_classes.long()], y_label[:, self.old_classes.long()])

                # Loss
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id)
                
                self.loss = loss.item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")

                # Update buffer
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())
                if self.params.tsne and task_id == self.params.n_tasks - 1:
                    self.tsne()
                if (j == (len(dataloader) - 1)) and (j > 0):
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
                    y_pred = self.model(self.transform_train(batch_x))
                    loss = 1*F.cross_entropy(y_pred, batch_y.long())+1*lo1

                    # Loss
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                        scaler.scale(loss).backward()
                        scaler.step(self.optim)
                        scaler.update()
                        if self.params.measure_drift >=0 and task_id > 0:
                            self.measure_drift(task_id)
                    self.optim.zero_grad()
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    # Update buffer
                    self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                    # Plot to tensorboard
                    if (j == (len(dataloader) - 1)) and (j > 0):
                        lg.info(
                            f"Phase : task{task_id}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                        )
            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                    sim_matrix_r_pre = torch.matmul(simclr_r[:aug1_mem.shape[0]],pre_u.t())

                    loss_sim_r = self.Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
                    loss_sim_pre = self.Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    lo1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd

                    y_label = self.model(self.transform_train(mem_x))
                    y_label_pre = self.previous_model_ori(self.transform_train(mem_x))

                    loss = 1 * F.cross_entropy(y_label, mem_y) + lo1  + \
                        1 * F.mse_loss(y_label_pre[:, self.old_classes.long()], y_label[:, self.old_classes.long()])
                    # Loss
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):    
                        scaler.scale(loss).backward()
                        scaler.step(self.optim)
                        scaler.update()
                        if self.params.measure_drift >=0 and task_id > 0:
                            self.measure_drift(task_id)
                    
                    self.optim.zero_grad()
                    self.loss = loss.item()
                    print(f"Phase : task{task_id}, Loss {self.loss:.3f}  batch {j}", end="\r")

                    # Update buffer
                    self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                    if (j == (len(dataloader) - 1)) and (j > 0):
                        if self.params.tsne and task_id == self.params.n_tasks - 1:
                            self.tsne()
                        lg.info(
                            f"Phase : task{task_id}  batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                        )
    
    def forward_slow(self, imgs, labels, **kwargs):
        no_drift = kwargs.get('no_drift', False)
        if self.previous_model is None or no_drift:
            with torch.no_grad():
                self.model.eval()
                b = min(50, len(imgs))
                n_steps = (len(imgs) -1) // b
                new_f = []
                for i in range(n_steps+1):
                    nf = self.model(imgs[i*b:(i+1)*b].to(device), is_simclr=True)[0]
                    new_f.append(nf)
                return torch.cat(new_f, dim=0)
        else:
            with torch.no_grad():
                self.previous_model.eval()
                self.model.eval()
                b = min(50, len(imgs))
                n_steps = (len(imgs) -1) // b
                old_f = []
                new_f = []
                for i in range(n_steps+1):
                    im_temp = imgs[i*b:(i+1)*b]
                    try:
                        of = self.previous_model(im_temp.to(device), is_simclr=True)[0]
                        nf = self.model(im_temp.to(device), is_simclr=True)[0]

                        old_f.append(of)
                        new_f.append(nf)
                    except:
                        pass
                return torch.cat(old_f, dim=0), torch.cat(new_f, dim=0)
    
    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        i = 0
        if not self.params.drop_fc:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    logits = self.model(self.transform_test(inputs))
                        
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
                    features, _ = self.model(self.transform_test(inputs), is_simclr=True)
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
                    
    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: self.model.eval()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            if use_proj:
                _, mem_representations_b = self.model(mem_imgs_b, is_simclr=True)
            else:
                mem_representations_b, _ = self.model(mem_imgs_b, is_simclr=True)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels
    