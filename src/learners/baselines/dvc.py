"""adapted from https://github.com/YananGu/DVC
"""
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np


from src.learners.baselines.er import ERLearner
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import DVCNet, DVCNet_ImageNet
from src.utils.utils import get_device

device = get_device()

class DVCLearner(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers['mgi_reservoir'](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
                lr=self.params.learning_rate,
                tf=self.transform_train
            )
        # according to paper and code
        self.dl_weight = 2.0 if self.params.dataset == 'cifar10' else 4.0
    
    def load_model(self):
        if self.params.dataset == 'imagenet100':
            return DVCNet_ImageNet(
                nf=self.params.nf,
                n_classes=self.params.n_classes,
                n_units=128,
                has_mi_qnet=True,
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int
            ).to(device)
        else:
            return DVCNet(
                nf=self.params.nf,
                n_classes=self.params.n_classes,
                n_units=128,
                has_mi_qnet=True,
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int
            ).to(device)
    
    def load_criterion(self):
        return torch.nn.MSELoss()
    
    def cross_entropy(z, zt):
        # eps = np.finfo(float).eps
        Pz = F.softmax(z, dim=1)
        Pzt = F.softmax(zt, dim=1)
        # make sure no zero for log
        # Pz  [(Pz   < eps).data] = eps
        # Pzt [(Pzt  < eps).data] = eps
        return -(Pz * torch.log(Pzt)).mean()

    def agmax_loss(self, y, ytrue, dl_weight=1.0):
        z, zt, zzt,_ = y
        Pz = F.softmax(z, dim=1)
        Pzt = F.softmax(zt, dim=1)
        Pzzt = F.softmax(zzt, dim=1)

        dl_loss = nn.L1Loss()
        yy = torch.cat((Pz, Pzt))
        zz = torch.cat((Pzzt, Pzzt))
        dl = dl_loss(zz, yy)
        dl *= dl_weight

        # -1/3*(H(z) + H(zt) + H(z, zt)), H(x) = -E[log(x)]
        entropy = self.entropy_loss(Pz, Pzt, Pzzt)
        return entropy, dl
    
    def cross_entropy_loss(self, z, zt, ytrue, label_smoothing=0):
        zz = torch.cat((z, zt))
        yy = torch.cat((ytrue, ytrue))
        if label_smoothing > 0:
            # ce = LabelSmoothingCrossEntropy(label_smoothing)(zz, yy)
            pass
        else:
            ce = nn.CrossEntropyLoss()(zz, yy)
        return ce
    
    def batch_probability(self, Pz, Pzt, Pzzt):
        Pz = Pz.sum(dim=0)
        Pzt = Pzt.sum(dim=0)
        Pzzt = Pzzt.sum(dim=0)

        Pz = Pz / Pz.sum()
        Pzt = Pzt / Pzt.sum()
        Pzzt = Pzzt / Pzzt.sum()

        # return Pz, Pzt, Pzzt
        return self.clamp_to_eps(Pz, Pzt, Pzzt)

    def clamp_to_eps(self, Pz, Pzt, Pzzt):
        eps = np.finfo(float).eps
        # make sure no zero for log
        Pz[(Pz < eps).data] = eps
        Pzt[(Pzt < eps).data] = eps
        Pzzt[(Pzzt < eps).data] = eps

        return Pz, Pzt, Pzzt
    
    def entropy_loss(self, Pz, Pzt, Pzzt):
        # negative entropy loss
        Pz, Pzt, Pzzt = self.batch_probability(Pz, Pzt, Pzzt)
        entropy = (Pz * torch.log(Pz)).sum()
        entropy += (Pzt * torch.log(Pzt)).sum()
        entropy += (Pzzt * torch.log(Pzzt)).sum()
        entropy /= 3
        return entropy
    
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
                self.loss = loss.mean().item()
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
        i = 0
        self.model.eval()
        if not self.params.drop_fc:
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(device)
                    inputs = self.transform_test(inputs)
                    logits = self.model(inputs, inputs)[0]
                    preds = logits.argmax(dim=1)

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
                    features = self.model.backbone(self.transform_test(inputs))[0]
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
            
    
    def forward_slow(self, imgs, labels):
        if self.previous_model is None:
            with torch.no_grad():
                self.model.eval()
                b = min(50, len(imgs))
                n_steps = (len(imgs) -1) // b
                new_f = []
                for i in range(n_steps+1):
                    nf = self.model.backbone(imgs[i*b:(i+1)*b].to(device))[0]

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
                    of = self.previous_model.backbone(imgs[i*b:(i+1)*b].to(device))[0]
                    nf = self.model.backbone(imgs[i*b:(i+1)*b].to(device))[0]

                    old_f.append(of)
                    new_f.append(nf)
                    
                return torch.cat(old_f, dim=0), torch.cat(new_f, dim=0)

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
                mem_representations_b = self.model(mem_imgs_b, mem_imgs_b)[0]
            else:
                mem_representations_b = self.model.backbone(mem_imgs_b)[0]
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels
    