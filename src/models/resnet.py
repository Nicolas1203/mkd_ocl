"""Code adapted from https://github.com/RaptorMai/online-continual-learning/models/resnet.py
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import logging as lg
import random as r

from easydict import EasyDict as edict
from torch.nn.functional import relu, avg_pool2d
from scipy import optimize

from src.utils.utils import make_orthogonal
from src.models.resnet_sdp import ResNet as sdp_resnet
if torch.cuda.is_available():
    dev = "cuda:0"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"
device = torch.device(dev)

EPSILON = 1e-10

bn = True

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if bn else nn.Identity()
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3,
                 ret_feat=True):
        super(ResNet, self).__init__()
        self.ret_feat = ret_feat
        self.in_planes = nf
        self.conv1 = conv3x3(input_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.instance_norm=instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(dim_in, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def logits(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        if self.ret_feat:
            return out
        else:
            return out, self.linear(out)


class SDResNet(nn.Module):
    """Adapted from https://github.com/luanyunteng/pytorch-be-your-own-teacher/blob/master/models/resnet.py#L142
    """
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super().__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(input_channels, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.instance_norm=instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(dim_in, num_classes, bias=bias)
        
        self.bottleneck1_1 = self.branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)
        
        self.downsample2_1 = nn.Sequential(
                            self.conv1x1(128 * block.expansion, 512 * block.expansion, stride=4),
                            nn.BatchNorm2d(512 * block.expansion),
            )
        self.bottleneck2_1 = self.branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)
        
        self.downsample3_1 = nn.Sequential(
                            self.conv1x1(256 * block.expansion, 512 * block.expansion, stride=2),
                            nn.BatchNorm2d(512 * block.expansion),
        )
        self.bottleneck3_1 = self.branchBottleNeck(256 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

    def conv1x1(self, in_planes, planes, stride=1):
        return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def branchBottleNeck(self, channel_in, channel_out, kernel_size):
        middle_channel = channel_out//4
        return nn.Sequential(
            nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            
            nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            
            nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            )
        
    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def logits(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x = self.layer2(x)
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x = self.layer3(x)
        middle_output3 = self.bottleneck3_1(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        x = self.layer4(x)
        # x = self.avgpool(x)
        x = avg_pool2d(x, 4)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return middle_output1, middle_output2, middle_output3, x, middle1_fea, middle2_fea, middle3_fea, final_fea


def SDResNet18(nclasses, dim_in=512, nf=64, bias=True):
    return SDResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)
    

def Reduced_ResNet18(nclasses, nf=20, bias=True, input_channels=3, instance_norm=False):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(
        BasicBlock, [2, 2, 2, 2],
        nclasses,
        nf,
        bias,
        input_channels=input_channels,
        instance_norm=instance_norm
        )

def ResNet18(nclasses, dim_in=512, nf=64, bias=True, ret_feat=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in, ret_feat=ret_feat)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


class SiamNet(nn.Module):
    def __init__(self, dim_in=160, proj_dim=128, input_channels=3):
        super().__init__()
        rep_dim=2048
        self.encoder = nn.Sequential(
            Reduced_ResNet18(100, input_channels=input_channels),
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_in, bias=False),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, rep_dim),
            nn.BatchNorm1d(rep_dim)
        )
        
        self.head1 = nn.Sequential(
            nn.Linear(rep_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, rep_dim)
        )

        self.head2 = nn.Sequential(
            nn.Linear(rep_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, rep_dim)
        )

    def forward(self, x, **kwargs):
        head_id = kwargs.get('head_id', 1)
        f = self.encoder(x)
        if head_id == 1:
            proj = self.head1(f)
        elif head_id == 2:
            proj = self.head2(f)

        return F.normalize(f, dim=1), F.normalize(proj, dim=1)
        # return f, proj


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    Original code from : https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py#L11
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        base_encoder = models.__dict__[base_encoder]
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, pretrained=False)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward_2v(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        return p1, p2, z1.detach(), z2.detach()
    
    def forward(self, x1):
        """
        Input:
            x1: images
        Output:
            p1, z1 : predictors and targets of the network
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC

        p1 = self.predictor(z1) # NxC

        p1 = F.normalize(p1, dim=1)
        z1 = F.normalize(z1, dim=1)

        return z1, p1


class BoostedResNet(nn.Module):
    def __init__(self, dim_in=160, **kwargs):
        super().__init__()
        # encoder params
        input_channels =  kwargs.get('input_channels', 3)
        proj_dim = kwargs.get('proj_dim', 128)
        self.encoder = Reduced_ResNet18(100, input_channels=input_channels)
        self.rep_dim = dim_in
        # Boosting params
        self.min_improvement = kwargs.get('min_improvement', 1e-6)
        self.max_boosting_iter = kwargs.get('max_boosting_iter', 1)
        self.no_head = kwargs.get('no_head', False)
        self.head = torch.Tensor([]).to(device)  # L matrix
        self.projector = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, proj_dim)
            )

    def reset_head(self):
        # self.head = torch.eye(self.rep_dim).to(device)
        if self.head.dim() > 1 and self.head.size(0) > self.head.size(1):
            eigenvalues, eigenvectors = torch.linalg.eigh(self.head.T@self.head)
            eigenvalues[eigenvalues < 0] = 0
            self.head = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        else:
            pass
            # self.head = torch.eye(self.rep_dim).to(device)

    def forward(self, x, **kwargs):
        feat = self.encoder.features(x)
        proj = self.projector(feat)

        return F.normalize(feat, dim=1), F.normalize(proj)

    # def forward(self, x, **kwargs):
    #     coef = kwargs.get('coef', 1)
    #     feat = self.encoder.features(x)
    #     if self.no_head:
    #         return feat, feat
    #     feat = F.normalize(feat, dim=1) * coef
    #     if self.head.size(0) > 0: 
    #         proj = feat @ self.head.T
    #     else:
    #         proj = feat

    #     return feat, proj
    
    def compute_weights(self, x, pairs):
        """Compute boosting weights for each pair of samples.

        Args:
            x (torch.Tensor): Input samples
            pairs (torch.Tensor): indexes of sample pairs

        Returns:
            tuple: positive weights, negative weights
        """
        pos_pairs, neg_pairs = pairs
        if self.head.size(0) == 0:
            pos_weight = (torch.ones((pos_pairs.size(0),))/pos_pairs.size(0)).to(device)
            neg_weight = (torch.ones((neg_pairs.size(0),))/neg_pairs.size(0)).to(device)
            W = torch.Tensor([1]).to(device)
        else:
            pos_delta = x[pos_pairs[:, 0],:] - x[pos_pairs[:, 1],:]
            neg_delta = x[neg_pairs[:, 0],:] - x[neg_pairs[:, 1],:]

            pos_scale = torch.max(torch.sum(torch.pow(pos_delta, 2), dim=1))
            neg_scale = torch.max(torch.sum(torch.pow(neg_delta, 2), dim=1))

            pos_delta = pos_delta/torch.sqrt(torch.max(pos_scale, neg_scale))
            neg_delta = neg_delta/torch.sqrt(torch.max(pos_scale, neg_scale))

            # XpL = pos_delta.view(-1, pos_delta.size(2)) @ self.head
            # XnL = neg_delta.view(-1, neg_delta.size(2)) @ self.head

            XpL = pos_delta @ self.head.T
            XnL = neg_delta @ self.head.T

            pos_dist = torch.pow(XpL, 2).sum(dim=1)
            neg_dist = torch.pow(XnL, 2).sum(dim=1)

            def func(alpha, pos_dist, neg_dist):
                r = torch.sum(torch.exp(alpha*pos_dist))
                r = r * torch.sum(torch.exp(-alpha*neg_dist))
                return r.cpu().numpy()
            res = optimize.minimize_scalar(
                    func,
                    args=(
                            pos_dist,  
                            neg_dist  
                        )
                    )
            lg.debug(f"Alpha identité {res.x}")
            if(res.x <= 0):
              alpha = torch.Tensor([1.0]).to(device)
            else:
              alpha = torch.Tensor([res.x]).to(device)
            
            self.head = self.head * torch.sqrt(alpha)

            pos_weight = torch.exp(alpha*pos_dist)
            neg_weight = torch.exp(-alpha*neg_dist)

            W = torch.mean(pos_weight)*torch.mean(neg_weight)

            pos_weight = pos_weight/torch.sum(pos_weight)
            pos_weight[pos_weight<1e-12] = 1e-12
            neg_weight = neg_weight/torch.sum(neg_weight)
            neg_weight[neg_weight<1e-12] = 1e-12

        return pos_weight, neg_weight, W
        
    def train_head(self, x, pairs, **kwargs):
        """Train the head of the model.
            TODO : add support partially labeled data
        Args:
            x (torch.Tensor): Encoded data pairs (bsz, nviews, rep_dim)
            pairs (torch.Tensor): Indices of positive and negative pairs
        """
        # get pairs
        pos_pairs, neg_pairs = pairs
        
        # reshape data from (bsz, nviews, rep_dim) to (bsz*nviews, rep_dim)
        x = torch.cat(torch.unbind(x, dim=1), dim=0)
        
        # Reset projection head
        self.reset_head()

        # Update data pairs weights
        pos_weight, neg_weight, W = self.compute_weights(x, (pos_pairs, neg_pairs))
        pos_delta = x[pos_pairs[:, 0],:] - x[pos_pairs[:, 1],:]
        neg_delta = x[neg_pairs[:, 0],:] - x[neg_pairs[:, 1],:]
        lg.debug(f"first W {W}")
        alpha = torch.Tensor([0]).to(device)
        for t in range(self.max_boosting_iter):
            A = (neg_delta*neg_weight[:,None]).T@neg_delta
            A = A - (pos_delta*pos_weight[:,None]).T@pos_delta
            try:
                eigenvalues, z = torch.linalg.eigh(A)
                # lg.debug(eigenvalues)
                z = z[None,:,-1]
            except:
                break
            pos_dist = torch.pow(pos_delta@z.T, 2).view(-1)
            neg_dist = torch.pow(neg_delta@z.T, 2).view(-1)

            def func(alpha, pos_weight, neg_weight, pos_dist, neg_dist):
                r = torch.sum(pos_weight*torch.exp(alpha*pos_dist))
                r = r * torch.sum(neg_weight*torch.exp(-alpha*neg_dist))
                return r.cpu().numpy()
            res = optimize.minimize_scalar(
                    func, 
                    args=(
                            pos_weight, 
                            neg_weight,
                            pos_dist,  
                            neg_dist  
                        )
                    )
            alpha = torch.Tensor([res.x]).to(device)
            lg.debug(f"eigenvalues {eigenvalues[-1].item()} alpha {alpha.item()}")
            if alpha.item() <= 0:
                break
            # alpha[alpha >= 100] = 100
            pos_weight = pos_weight*torch.exp(alpha*pos_dist)
            neg_weight = neg_weight*torch.exp(-alpha*neg_dist)
            w = torch.sum(pos_weight)*torch.sum(neg_weight)

            if w >= 1:
                break
            if self.head.dim() > 1:
                self.head = torch.vstack([self.head, torch.sqrt(alpha)*z.to(device)])
            else:
                self.head = torch.sqrt(alpha)*z.to(device).view(1, -1)
            W = W * w
            lg.debug(f"W {W.item()} w {w.item()}")
            pos_weight = pos_weight/torch.sum(pos_weight)
            pos_weight[pos_weight<1e-12] = 1e-12
            neg_weight = neg_weight/torch.sum(neg_weight)
            neg_weight[neg_weight<1e-12] = 1e-12
        lg.debug(f"last W {W.item()}")
        # self.head = self.head.T
        return pos_weight, neg_weight, alpha.item()


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512,
                input_channels=3, instance_norm=False, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf)
        self.ortho = head == 'ortho'
        if head == 'linear':
            self.head = nn.Linear(dim_in, proj_dim)
        elif head == 'mlp' or self.ortho:
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_int),
                nn.ReLU(inplace=True),
                nn.Linear(dim_int, proj_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, **kwargs):
        feat = self.encoder(x)
        proj = self.head(feat) if self.head is not None else feat

        feat_norm = kwargs.get('feat_norm', True)
        proj_norm = kwargs.get('proj_norm', True)

        y = kwargs.get('y')
             
        if self.ortho and y is not None:
            feat = make_orthogonal(feat.double(), y.long()).float()
            proj = self.head(feat) if self.head is not None else feat
            
        feat = F.normalize(feat, dim=1) if feat_norm else feat
        proj = F.normalize(proj, dim=1) if proj_norm else proj

        return feat, proj
    

class SupConResNet_ImageNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512,
                input_channels=3, instance_norm=False, nf=64):
        super().__init__()
        self.encoder = ImageNet_ResNet18(100, dim_in=dim_in, nf=nf)
        self.ortho = head == 'ortho'
        if head == 'linear':
            self.head = nn.Linear(dim_in, proj_dim)
        elif head == 'mlp' or self.ortho:
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_int),
                nn.ReLU(inplace=True),
                nn.Linear(dim_int, proj_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x, **kwargs):
        feat = self.encoder(x)
        proj = self.head(feat) if self.head is not None else feat

        feat_norm = kwargs.get('feat_norm', True)
        proj_norm = kwargs.get('proj_norm', True)

        y = kwargs.get('y')
             
        if self.ortho and y is not None:
            feat = make_orthogonal(feat.double(), y.long()).float()
            proj = self.head(feat) if self.head is not None else feat
            
        feat = F.normalize(feat, dim=1) if feat_norm else feat
        proj = F.normalize(proj, dim=1) if proj_norm else proj

        return feat, proj


class GMLResnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.sup_head = nn.Sequential(
            nn.Linear(dim_in, dim_int),
            nn.ReLU(inplace=True),
            nn.Linear(dim_int, n_classes)
        )
        self.unsup_head = nn.Sequential(
            nn.Linear(dim_in, dim_int),
            nn.ReLU(inplace=True),
            nn.Linear(dim_int, proj_dim)
        )
    
    def forward_double(self, x, **kwargs):
        feat = self.encoder(x)
        proj_sup = self.sup_head(feat)
        proj_unsup = self.unsup_head(feat)
        
        # proj_sup = F.normalize(proj_sup, dim=1)
        proj_unsup = F.normalize(proj_unsup, dim=1)

        return proj_unsup, proj_sup
    
    # def forward(self, x, **kwargs):
    #     feat = self.encoder(x)
    #     proj = self.unsup_head(feat)
        
    #     feat_norm = kwargs.get('feat_norm', True)
    #     proj_norm = kwargs.get('proj_norm', True)

    #     feat = F.normalize(feat, dim=1) if feat_norm else feat
    #     proj = F.normalize(proj, dim=1) if proj_norm else proj

    #     return feat, proj
    
    def forward(self, x, **kwargs):
        feat = self.encoder(x)
        logits = self.unsup_head(feat)
        
        return logits

class OCMResnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )
    
    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return out

class OCMResnet_ImageNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ImageNet_ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )
    
    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return out
    

class GSAResnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )
    
    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return out


class GSAResnet_ImageNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, head='mlp', proj_dim=128, dim_int=512, n_classes=10, nf=64):
        super().__init__()
        self.encoder = ImageNet_ResNet18(100, nf=nf, dim_in=dim_in)
        self.head_representation = nn.Sequential(
            nn.Linear(dim_in, proj_dim)
        )
        self.head_classification = nn.Sequential(
            nn.Linear(dim_in, n_classes)
        )
    
    def forward(self, x, is_simclr=False):

        features = self.encoder(x)

        if is_simclr:
            out = self.head_representation(features)
            return features, out
        else:
            out = self.head_classification(features)
            return out


class SimCLRResNet(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, projection_head=True, pretrained=False):
        super().__init__()
        base_encoder = eval(base_encoder)
        self.encoder = base_encoder(pretrained=pretrained)
        self.feature_dim = self.encoder.fc.in_features
        
        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        if projection_head:
            lg.debug(f"Representation dimension is : {self.feature_dim}")
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.feature_dim, projection_dim))
        else:
            self.projector = nn.Linear(self.feature_dim, projection_dim)

    def forward(self, x):
        feature = self.encoder(x)
        projection = self.projector(feature)
        projection = F.normalize(projection, dim=1)

        return feature, projection


class DVCNet(nn.Module):
    def __init__(self,
                 nf,
                 n_units,
                 n_classes,
                 has_mi_qnet=True,
                 **kwargs):
        super(DVCNet, self).__init__()
        dim_in = kwargs.get('dim_in')
        dim_int = kwargs.get('dim_int')
        self.backbone = SupConResNet(head='linear', proj_dim=n_classes, nf=nf, dim_in=dim_in, dim_int=dim_int)
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        fea,zz = self.backbone(xx)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]

class DVCNet_ImageNet(nn.Module):
    def __init__(self,
                 nf,
                 n_units,
                 n_classes,
                 has_mi_qnet=True,
                 **kwargs):
        super().__init__()
        dim_in = kwargs.get('dim_in')
        dim_int = kwargs.get('dim_int')
        self.backbone = SupConResNet_ImageNet(head='linear', proj_dim=n_classes, nf=nf, dim_in=dim_in, dim_int=dim_int)
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        fea,zz = self.backbone(xx)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]

class QNet(nn.Module):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )

    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt

class DVCNet_SDP(nn.Module):
    def __init__(self,
                 nf,
                 n_units,
                 n_classes,
                 has_mi_qnet=True,
                 **kwargs):
        super().__init__()
        opt = edict(
            {
                "depth": 18,
                "num_classes": n_classes,
                "in_channels": 3,
                "bn": True,
                "normtype": "BatchNorm",
                "activetype": "ReLU",
                "pooltype": "MaxPool2d",
                "preact": False,
                "affine_bn": True,
                "bn_eps": 1e-6,
                "compression": 0.5,
            }
        )
        self.backbone = sdp_resnet(opt)
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz,fea = self.backbone(xx, get_feature=True)

        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]


class BYOLResnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=512, proj_dim=128, dim_int=512, nf=64, **kwargs):
        super().__init__()
        self.encoder = ResNet18(100, nf=nf, dim_in=dim_in)
        self.projector = nn.Sequential(
                nn.Linear(dim_in, dim_int),
                nn.ReLU(inplace=True),
                nn.Linear(dim_int, proj_dim)
            )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, dim_int),
            nn.ReLU(inplace=True),
            nn.Linear(dim_int, proj_dim)
        )
    
    def forward(self, x, **kwargs):
        feat = self.encoder(x)
        pred = self.predictor(self.projector(feat))
        
        feat_norm = kwargs.get('feat_norm', True)
        proj_norm = kwargs.get('proj_norm', True)

        feat = F.normalize(feat, dim=1) if feat_norm else feat
        pred = F.normalize(pred, dim=1) if proj_norm else pred

        return feat, pred

    def forward_ema(self, x, **kwargs):
        return F.normalize(self.projector(self.encoder(x)), dim=1)

class ImageNet_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, dim_in=512, instance_norm=False, input_channels=3):
        super(ImageNet_ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = nn.Conv2d(input_channels, nf * 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nf * 1) if bn else nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.instance_norm=instance_norm
        if self.instance_norm:
            self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(dim_in, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out

    def forward(self, x):
        out = self.features(x)
        return out

    def full(self, x):
        out = self.features(x)
        logits = self.linear(out)
        return out, logits


def ImageNet_ResNet18(nclasses, dim_in=512, nf=64, bias=True):
    return ImageNet_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias, dim_in=dim_in)