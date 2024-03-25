import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from copy import deepcopy
from sklearn.manifold import TSNE

from src.learners.ce import BaseLearner
from src.utils.losses import SupConLoss, WKDLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device, filter_labels

device = get_device()

class ER_KDULearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        if self.params.drop_fc:
            self.init_results()
            
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
            alpha_kd=self.params.alpha_kd
        )
        
        self.previous_model = None

    def load_model(self, **kwargs):
        if self.params.dataset == 'imagenet100':
            model = ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            )
        else:
            model = ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf,
                ret_feat=not self.params.drop_fc
            )
        # model = nn.DataParallel(model)
        model.to(self.device)
        return model
    
    def load_criterion(self):
        return nn.CrossEntropyLoss()
    
    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id = kwargs.get("task_id", None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    combined_x = self.transform_train(combined_x)

                    # Inference
                    logits_stu = self.model.logits(combined_x)

                    # Loss
                    loss = self.criterion(logits_stu, combined_y.long())
                    
                    if self.previous_model is not None:
                        logits_tea = self.previous_model.logits(combined_x)
                        loss += self.params.kd_lambda * self.wkdloss(logits_tea.detach(), logits_stu)
                        
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                if self.params.tsne and task_id == self.params.n_tasks - 1:
                    self.tsne()
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
        if self.params.kdu:
            self.previous_model = deepcopy(self.model)

    def tsne(self, **kwargs):
        no_drift = kwargs.get('no_drift', False)
        
        plt.cla()
        mem_imgs, mem_labels = self.buffer.get_all()
        features = self.forward_slow(mem_imgs, mem_labels, no_drift=no_drift)
        np.save(f"tsne/features_{self.params.learner}.npy", features.cpu().numpy())
        np.save(f"tsne/labels_{self.params.learner}.npy", mem_labels.cpu().numpy())
        
    def measure_drift(self, task_id):
        if self.previous_model is None:
            self.previous_model = deepcopy(self.model)
        else:
            self.previous_model.eval()
            self.model.eval()
            mem_imgs, mem_labels = self.buffer.get_all()
            past_label = self.params.labels_order[:task_id*10]
            
            drifts_all = torch.FloatTensor(size=(0,))
            for i in range(len(past_label)):
                past_index = torch.nonzero(filter_labels(mem_labels, [past_label[i]])).flatten()
                img_past = mem_imgs[past_index]

                if len(img_past) > 2:
                    oldf, newf = self.forward_slow(img_past, past_label)
                    drift_c = torch.sqrt(((oldf - newf) ** 2).sum(1))
                    drifts_all = torch.cat([drifts_all.cpu(), drift_c.cpu()])
            self.drift.append(drifts_all.mean().item())
            self.model.train()
            self.previous_model = deepcopy(self.model)
            df = pd.DataFrame(self.drift)
            df.to_csv(f"./drifts/aaai24/{self.params.learner}_drift.csv")
    
    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]
    
    def evaluate(self, dataloaders, task_id):
        if not self.params.drop_fc:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model.eval()
                accs = []

                all_preds = []
                all_targets = []
                for j in range(task_id + 1):
                    test_preds, test_targets = self.encode(dataloaders[f"test{j}"])
                    acc = accuracy_score(test_targets, test_preds)
                    accs.append(acc)
                    # Wandb logs
                    if not self.params.no_wandb:
                        all_preds = np.concatenate([all_preds, test_preds])
                        all_targets = np.concatenate([all_targets, test_targets])
                        wandb.log({
                            f"acc_{j}": acc,
                            "task_id": task_id
                        })
                
                # Make confusion matrix
                if not self.params.no_wandb:
                    # re-index to have classes in task order
                    all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                    all_preds = [self.params.labels_order.index(int(i)) for i in all_preds]
                    n_im_pt = self.params.n_classes // self.params.n_tasks
                    all_targets_b = [0 if i < (task_id)*n_im_pt else 1 for i in all_targets]
                    all_preds_b = [0 if i < (task_id)*n_im_pt else 1 for i in all_preds]
                    cm = confusion_matrix(all_targets, all_preds)
                    mcc = matthews_corrcoef(all_targets, all_preds)
                    mcc_b = matthews_corrcoef(all_targets_b, all_preds_b)
                    cm_log = np.log(1 + cm)
                    fig = plt.matshow(cm_log)
                    wandb.log({
                            "cm_raw": cm,
                            "cm": fig,
                            "mcc": mcc,
                            "mcc_b": mcc_b,
                            "task_id": task_id
                        })
                    
                for _ in range(self.params.n_tasks - task_id - 1):
                    accs.append(np.nan)
                self.results.append(accs)
                
                line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
                line = line[0].to_numpy().tolist()
                self.results_forgetting.append(line)

                self.print_results(task_id)

                return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1])
        else:
            return super().evaluate(dataloaders, task_id)
            
    def print_results(self, task_id):
        if not self.params.drop_fc:
            n_dashes = 20
            pad_size = 8
            print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
            
            print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
            for line in self.results:
                print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
        else:
            super().print_results(task_id)
    
    def encode(self, dataloader, use_proj=False, nbatches=-1):
        if not self.params.drop_fc:
            i = 0
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(device)
                    logits = self.model.logits(self.transform_test(inputs))
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
            return super().encode(dataloader, use_proj)
    
    def save_results(self):
        if not self.params.drop_fc:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")
            print(f"Saving accuracy results in : {results_dir}")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            
            df_avg = pd.DataFrame()
            cols = [f'task {i}' for i in range(self.params.n_tasks)]
            # Loop over classifiers results
            # Each classifier has a value for every task. NaN if future task
            df_clf = pd.DataFrame(self.results, columns=cols)
            # Average accuracy over all tasks with not NaN value
            df_avg = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)
            df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
            
            df_avg = pd.DataFrame()
            print(f"Saving forgetting results in : {results_dir}")
            cols = [f'task {i}' for i in range(self.params.n_tasks)]
            df_clf = pd.DataFrame(self.results_forgetting, columns=cols)
            df_avg = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, 'forgetting.csv'), index=False)
            df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

            self.save_parameters()
        else:
            super().save_results()
    
    def save_results_offline(self):
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)

        print(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        pd.DataFrame(self.results).to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

        self.save_parameters()

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

    def forward_slow(self, imgs, labels, **kwargs):
        if self.previous_model is None:
            with torch.no_grad():
                self.model.eval()
                b = min(50, len(imgs))
                n_steps = (len(imgs) -1) // b
                new_f = []
                for i in range(n_steps+1):
                    nf = self.model.features(imgs[i*b:(i+1)*b].to(device))

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
                    of = self.previous_model.features(imgs[i*b:(i+1)*b].to(device))
                    nf = self.model.features(imgs[i*b:(i+1)*b].to(device))

                    old_f.append(of)
                    new_f.append(nf)
                    
                return torch.cat(old_f, dim=0), torch.cat(new_f, dim=0)