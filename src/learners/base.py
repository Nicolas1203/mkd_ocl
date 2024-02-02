from random import random
import torch
import logging as lg
import os
import pickle
import time
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch.nn as nn
import random as r
import json
import torchvision
import wandb

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from torchvision import transforms
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from copy import deepcopy

from src.utils.utils import save_model
from src.models import resnet
from src.utils.metrics import forgetting_line 
from src.utils.utils import get_device

device = get_device()

class BaseLearner(torch.nn.Module):
    def __init__(self, args):
        """Learner abstract class
        Args:
            args: argparser arguments
        """
        super().__init__()
        self.params = args
        self.device = get_device()
        self.init_tag()
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.buffer = None
        self.start = time.time()
        self.criterion = self.load_criterion()
        if self.params.tensorboard:
            self.writer = self.load_writer()
        # self.classifiers_list = ['logreg', 'knn', 'linear', 'svm', 'ncm']  # Classifiers used for evaluating representation
        self.classifiers_list = ['ncm']  # Classifiers used for evaluating representation
        self.loss = 0
        self.stream_idx = 0
        self.init_results()
        
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.params.dataset == 'tiny' else nn.Identity()
        normalize = nn.Identity()
        if self.params.tf_type == 'partial':
            self.transform_train = nn.Sequential(
                torchvision.transforms.RandomCrop(self.params.img_size, padding=4),
                RandomHorizontalFlip(),
                normalize
            ).to(device)
        elif self.params.tf_type == 'full':
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)
        
        self.transform_test = nn.Sequential(
            normalize
        ).to(device)

    def init_results(self):
        """Initialise results placeholder
        """
        self.results = {}
        self.results_forgetting = {}
        for clf_name in self.classifiers_list:
            self.results[clf_name] = []
            self.results_forgetting[clf_name] = []

    def init_tag(self):
        """Initialise tag for experiment
        """
        if self.params.training_type == 'inc':
            self.params.tag = f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}{self.params.tag}"
        elif self.params.training_type == "blurry":
            self.params.tag =\
                 f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}blurry{self.params.blurry_scale}{self.params.tag}"
        else:
            self.params.tag = f"{self.params.learner},{self.params.dataset},{self.params.epochs}b{self.params.batch_size},uni{self.params.tag}"
        print(f"Using the following tag for this experiment : {self.params.tag}")
    
    def load_writer(self):
        """Initialize tensorboard summary writer
        """
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(self.params.tb_root, self.params.tag))
        return writer

    def save(self, model_name):
        """Save model and buffer.
        Args:
            model_name (str): Name of the model file
        """
        if self.params.save_ckpt:
            save_dir = os.path.join(self.params.ckpt_root, self.params.tag, str(self.params.run_id))
            save_model(
                self.model.state_dict(),
                model_name,
                dir=save_dir
                )
            if self.buffer is not None:
                lg.debug("Saving memory buffer")
                with open(os.path.join(save_dir, f"memory_{self.buffer.n_seen_so_far}.pkl"), 'wb') as memory_file:
                    pickle.dump(self.buffer, memory_file)
    
    def resume(self, model_path=None, buffer_path=None):
        """Resume model and buffer.
        Args:
            model_path (str, optional): Path to the model to load. Defaults to None.
            buffer_path (str, optional): Path to the buffer to load. Defaults to None.
        """
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        if buffer_path is not None:
            with open(buffer_path, 'rb') as f:
                self.buffer = pickle.load(f)
            f.close()
        if model_path is None or buffer_path is None:
            raise Warning("Invalid model and/or buffer state for resume or test argument.")
        torch.cuda.empty_cache()

    def load_model(self):
        """Load model
        Returns:
            untrained torch backbone model
        """
        model = resnet.SupConResNet(
            head=self.params.head,
            dim_in=self.params.dim_in,
            dim_int=self.params.dim_int,
            proj_dim=self.params.proj_dim,
            input_channels=self.params.nb_channels,
            instance_norm=self.params.instance_norm,
            nf=self.params.nf
        )
        model = nn.DataParallel(model)
        model.to(device)
        return model

    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                )
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_scheduler(self):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError
    
    def train(self, dataloader, task, **kwargs):
        raise NotImplementedError

    def evaluate(self, dataloaders, task_id, **kwargs):
        """Evaluate the model after training ona specified task
        Args:
            dataloaders (Dictonnary of dataloaders): Dictionnary containing every train and test loader for each task
            task_id (int): id of the current task 
        Returns:
            NCM current average accuracy and average forgetting
        """
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            if self.params.eval_mem:
                mem_representations, mem_labels = self.get_mem_rep_labels(use_proj=self.params.eval_proj)
            else:
                mem_representations, mem_labels = self.sample_rep_label(
                    dataloader=dataloaders['train'],
                    labels_list=self.params.labels_order[:(task_id+1)*step_size],
                    data_per_class=self.params.lab_pc,
                    output_format="torch"
                )
            classifiers = self.init_classifiers()
            classifiers = self.fit_classifiers(classifiers=classifiers, representations=mem_representations, labels=mem_labels)
            
            accs = {}
            representations = {}
            targets = {}
            # Init representations - labels pairs
            for j in range(task_id + 1):
                test_representation, test_targets = self.encode(dataloaders[f"test{j}"], use_proj=self.params.eval_proj)
                representations[f"test{j}"] = test_representation
                targets[f"test{j}"] = test_targets
            # Compute accuracies
            for clf, clf_name in zip(classifiers, self.classifiers_list):
                accs[clf_name] = []
                for j in range(task_id + 1):
                    acc = accuracy_score(targets[f"test{j}"], clf.predict(representations[f'test{j}'])) 
                    accs[clf_name].append(acc)
                    # Wandb logs
                    if not self.params.no_wandb:
                        wandb.log({
                            f"{clf_name}_{j}": acc,
                            "task_id": task_id
                        })
            for clf_name in accs:
                for j in range(self.params.n_tasks - task_id - 1):
                    accs[clf_name].append(np.nan)
                self.results[clf_name].append(accs[clf_name])
            # Compute forgettings
            for clf_name in self.results:
                line = forgetting_line(pd.DataFrame(self.results[clf_name]), task_id=task_id, n_tasks=self.params.n_tasks)
                line = line[0].to_numpy().tolist()
                self.results_forgetting[clf_name].append(line)
            
            self.print_results(task_id)

            return np.nanmean(self.results['ncm'][-1]), np.nanmean(self.results_forgetting['ncm'][-1])

    def evaluate_offline(self, dataloaders, epoch):
        """Evaluate the model on test set at a specified epoch. Train loader is used for training
        a classifier on some random images selected from train set
        Args:
            dataloaders (dict): Dictionnary containing dataloaders 'train' and 'test'
        """
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            if self.params.eval_mem:
                mem_representations, mem_labels = self.get_mem_rep_labels()
            else:
                mem_representations, mem_labels = self.sample_rep_label(
                    dataloader=dataloaders['train'],
                    labels_list=self.params.labels_order,
                    data_per_class=self.params.lab_pc,
                    output_format="torch"
                )
            classifiers = self.init_classifiers()
            classifiers = self.fit_classifiers(
                classifiers=classifiers,
                representations=mem_representations,
                labels=mem_labels
            )

            test_representations, test_targets = self.encode(dataloaders['test'])

            for clf, clf_name in zip(classifiers, self.classifiers_list):
                acc = accuracy_score(test_targets, clf.predict(test_representations))
                self.results[clf_name].append(acc)
            
            if not ((epoch + 1) % 5):
                self.save(model_name=f"{self.params.tag}_{epoch}.pth")
        
        lg.info(f"ACCURACY {self.results['ncm'][-1]}")
        return self.results['ncm'][-1]

    def sample_rep_label(self, dataloader, labels_list, data_per_class, output_format='torch', return_proj=False, return_raw=False):
        """Sample labeled data randomly from dataloader. Returns data representation and labels such that every 
        label is equally present.
        Exemple : labels_list = [0,1], data_per_class = 3
        returns labels [1,1,0,1,0,0] and reps of shape (6, rep_dim)
        Args:
            dataloader (torch dataloader):      Dataloader for iterating over labeled data 
            labels_list (list):                 List of requested labels
            data_per_class (_type_):            Number of data per class
            output_format (str, optional):      Datatype fo the output image. Defaults to 'torch'
            return_proj (bool, optional):       Whether to return the projection of the representation of the model
            return_raw (bool, optional)         Whether to return the raw data instead of the projection
        Returns:
            out_rep   (numpy array or torch tensor):     List of representations
            out_labels (numpy array):                     List of labels
        """
        out_rep = []
        out_labels = []
        for sample in dataloader:
            imgs, labels = sample[0], sample[1]
            labels = np.array(labels)
            # Loop over batch
            for img, label in zip(imgs, labels):
                if np.isin(label, labels_list):
                    nb_data_per_label = np.bincount(out_labels, minlength=np.max(labels_list)+1)
                    if nb_data_per_label[int(label)] < data_per_class:
                        if return_raw:
                            out_rep.append(img.cpu().numpy())
                        else:
                            i = 1 if return_proj else 0
                            out_rep.append(np.array(self.model(self.transform_test(img.to(self.device).unsqueeze(0)))[i][0].cpu()))
                        out_labels.append(label)
                    if len(out_labels) >= (len(labels_list) * data_per_class):
                        if output_format == "numpy":
                            if return_raw:
                                temp = np.array(out_rep)
                                out_rep = np.ascontiguousarray(np.swapaxes(np.swapaxes(temp, 1, 2), 2, 3)*255)
                            out_labels = np.array(out_labels)
                        elif output_format == "torch":
                            out_rep = torch.Tensor(np.array(out_rep))
                            out_labels = torch.Tensor(np.array(out_labels))
                        return out_rep, out_labels

    def encode(self, dataloader, use_proj=False, nbatches=-1):
        """Compute representations - labels pairs for a certain number of batches of dataloader.
            Not really optimized.
        Args:
            dataloader (torch dataloader): dataloader to encode
            nbatches (int, optional): Number of batches to encode. Use -1 for all. Defaults to -1.
        Returns:
            representations - labels pairs
        """
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                if use_proj:
                    _, features = self.model(self.transform_test(inputs))
                else:
                    features, _ = self.model(self.transform_test(inputs))
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = features.cpu().numpy()
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.vstack([all_feat, features.cpu().numpy()])
                i += 1
        return all_feat, all_labels

    def print_results(self, task_id):
        """Log learner's results as accuracy and forgetting
        Example : 
                root - INFO - --------------------FORGETTING--------------------
                root - INFO - ncm     0.0000   nan      nan      nan      nan      0.0000
                root - INFO - ncm     0.2885   0.0000   nan      nan      nan      0.2885
                root - INFO - ncm     0.2935   0.2225   0.0000   nan      nan      0.2580
                root - INFO - ncm     0.4615   0.3190   0.0370   0.0000   nan      0.2725
                root - INFO - ncm     0.5815   0.2155   0.1795   0.0250   0.0000   0.2504
                root - INFO - --------------------ACCURACY--------------------
                root - INFO - ncm     0.7750   nan      nan      nan      nan      0.7750
                root - INFO - ncm     0.4865   0.5260   nan      nan      nan      0.5062
                root - INFO - ncm     0.4815   0.3035   0.5150   nan      nan      0.4333
                root - INFO - ncm     0.3135   0.2070   0.4780   0.2875   nan      0.3215
                root - INFO - ncm     0.1935   0.3105   0.3355   0.2625   0.3045   0.2813
        Args:
            task_id (int):      Number of the task where the learner was just trained on.
        """
        n_dashes = 20
        pad_size = 8
        lg.info('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        lg.info('-' * n_dashes + "FORGETTING" + '-' * n_dashes)
        for clf_name in self.results:
            for line in self.results_forgetting[clf_name]:
                lg.info(f'{clf_name}'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line) + f" {np.nanmean(line):.4f}")
        
        lg.info('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for clf_name in self.results:
            for line in self.results[clf_name]:
                lg.info(f'{clf_name}'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line) + f" {np.nanmean(line):.4f}")

    def save_results(self):
        """Save results as csv.
        """
        results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")
        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        df_avg = pd.DataFrame()
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        # Loop over classifiers results
        for clf_name in self.results:
            # Each classifier has a value for every task. NaN if future task
            df_clf = pd.DataFrame(self.results[clf_name], columns=cols)
            # Average accuracy over all tasks with value not NaN
            df_avg[clf_name] = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, f'{clf_name}.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
        
        df_avg = pd.DataFrame()
        lg.info(f"Saving forgetting results in : {results_dir}")
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        for clf_name in self.results_forgetting:
            df_clf = pd.DataFrame(self.results_forgetting[clf_name], columns=cols)
            df_avg[clf_name] = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, f'{clf_name}_forgetting.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

        self.save_parameters()
    
    def save_results_offline(self):
        """Save results as csv for offline training
        """
        results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")

        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        df_all = pd.DataFrame()
        for clf_name in self.results:
            df_all[clf_name] = pd.DataFrame(self.results[clf_name])
        df_all.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)
        self.save_parameters()

    def save_parameters(self):
        """Save parameters used for training
        """
        filename = os.path.join(self.params.results_root, self.params.tag, f'run{self.params.seed}/params_used.json')
        with open(filename, 'w') as f:
            json.dump(self.params.__dict__, f, indent=2)

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
                _, mem_representations_b = self.model(mem_imgs_b)
            else:
                mem_representations_b, _ = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels

    def init_classifiers(self):
        """Initiliaze every classifier for representation transfer learning
        Returns:
            List of initialized classifiers
        """
        # logreg = LogisticRegression(random_state=self.params.seed)
        # knn = KNeighborsClassifier(3)
        # linear = MLPClassifier(hidden_layer_sizes=(200), activation='identity', max_iter=500, random_state=self.params.seed)
        # svm = SVC()
        ncm = NearestCentroid()
        # return [logreg, knn, linear, svm, ncm]
        return [ncm]
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit_classifiers(self, classifiers, representations, labels):
        """Fit every classifiers on representation - labels pairs
        Args:
            classifiers : List of sklearn classifiers
            representations (torch.Tensor): data representations 
            labels (torch.Tensor): data labels
        Returns:
            List of trained classifiers
        """
        for clf in classifiers:
            clf.fit(representations.cpu(), labels)
        return classifiers

    def after_eval(self, **kwargs):
        pass

    def before_eval(self, **kwargs):
        if self.params.review:
            model_backup = deepcopy(self.model)
            mem_x, mem_y = self.buffer.get_all()

            for e in range(self.mem_epochs):
                idx = np.random.permutation(len(mem_x)).tolist()
                mem_x = mem_x[idx]
                mem_y = mem_y[idx]
                self.model = self.model.train()
                batch_size = 64
                
                for j in range(len(mem_y) // batch_size):
                    batch_x = mem_x[batch_size * j:batch_size * (j + 1)].to(self.device)
                    aug1, aug2 = self.transform_train(batch_x), self.transform_train(batch_x)
                    _, p1 = self.model(aug1)
                    _, p2 = self.model(aug2)
                    p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                    loss = self.criterion(p, labels=mem_y[batch_size * j:batch_size * (j + 1)].to(self.device))
                    self.loss = loss.item()

                    # Optim
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.optim.step()

            self.model = model_backup

    def augment(self, combined_x, mem_x, batch_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for key in self.tf_seq:
                if 'aug' in key:
                    augmentations.append(self.tf_seq[key](combined_x))
                else:
                    n_repeat = self.get_n_repeat(key)
                    batch1 = combined_x
                    batch2 = batch_x if self.params.daa else combined_x
                    for _ in range(n_repeat):
                        augmentations.append(self.tf_seq[key](batch1, batch2, model=self.model))
            augmentations.append(combined_x)
            return augmentations

    def train_inc(self, **kwargs):
        raise NotImplementedError

    def train_blurry(self, **kwargs):
        raise NotImplementedError