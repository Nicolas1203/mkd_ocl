import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ImageNet(ImageFolder):
    def __init__(self, root, train, transform):
        super().__init__(os.path.join(root, 'imagenet/train' if train else 'imagenet/val'), transform)
        self.transform_in = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
        ])
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.transform_in(img)
        label = torch.tensor(label)
        return img, label