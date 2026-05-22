import os
from os.path import isdir
import numpy as np
from pickle import TRUE
import tarfile
import wget
from pathlib import Path
from PIL import Image
import cv2

from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

DATASETS_PATH = Path("./datasets")
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])

class MVTecDataset:
    def __init__(self, cls : str, size : int = 224, dataset_dir: str = './datasets', resize_method: str = 'transform'):
        self.cls = cls
        self.size = size
        self.pin_memory = False
        #训练集和测试集使用的图像缩放方法必须一致，使用中发现如果训练集用 transform, 测试集用cv2，会出现测试效果很差的情况
        self.train_ds = MVTecTrainDataset(cls, size, dataset_dir=dataset_dir, resize_method=resize_method)
        self.test_ds = MVTecTestDataset(cls, size, dataset_dir=dataset_dir, resize_method=resize_method)

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds,num_workers=2,pin_memory=self.pin_memory), DataLoader(self.test_ds,num_workers=1,pin_memory=self.pin_memory)

class Cv2AdaptiveResize(object):
    def __init__(self, size : int, interpolation : int = cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self, pil_img):
        # img是一个pil的图片格式，如果进行数据处理，就统一数据类型
        img = np.asarray(pil_img)
        width, height = pil_img.size
        
        if isinstance(self.size, int):
            new_height = self.size
            new_width = int(width * new_height / height)
            new_width = int(new_width / 4) * 4  #宽度确保4的倍数
        elif isinstance(self.size, list) and len(self.size) == 2:
            new_width, new_height = self.size
        else:
            raise ValueError(f'Invalid self.size: {self.size}')
        '''
        for x in img:
            for y in x:
                print(y)
        print('-------------')
        '''        
        img = cv2.resize(img, (new_width, new_height), interpolation=self.interpolation)
        return Image.fromarray(img)

class TransformAdaptiveResize(object):
    def __init__(self, size : int, interpolation : int = transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
        
    def __call__(self,pil_img):
        width,height = pil_img.size
        new_height = self.size
        new_width = int(width * new_height/height)
        new_width = int(new_width/4)*4 #宽度确保4的倍数
        return F.resize(pil_img, (new_height,new_width), self.interpolation, self.max_size, self.antialias)
        
class MVTecTrainDataset(ImageFolder):
    def __init__(self, cls : str, size : int, dataset_dir: str = './datasets', resize_method: str = 'transform'):
        if resize_method == 'transform':
            resizeClass =  TransformAdaptiveResize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        else:
            resizeClass =  Cv2AdaptiveResize(size, interpolation=cv2.INTER_LINEAR)
        super().__init__(
            root= Path(dataset_dir) / cls / "train",
            transform=transforms.Compose([
                resizeClass,
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        self.cls = cls
        self.size = size

        
# 使用transforms.resize 进行图像尺寸变换
class MVTecTestDataset(ImageFolder): 
    def __init__(self, cls : str, size : int, dataset_dir: str = './datasets', resize_method: str = 'transform'):
        if resize_method == 'transform':
            resizeClass =  TransformAdaptiveResize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        else:
            resizeClass =  Cv2AdaptiveResize(size, interpolation=cv2.INTER_LINEAR)
        print('dataset_dir',dataset_dir)
        print('cls',cls)
        
        if isinstance(size, int):
            pass
        elif isinstance(size, list) and len(size) == 2:
            size = size[::-1]
        else:
            raise ValueError(f'Invalid size: {size}')
        print(f'MVTecTestDataset: size={size}')
            
        super().__init__(
            root= Path(dataset_dir) / cls / "test",
            transform=transforms.Compose([
                resizeClass,
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            target_transform=transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]),
        )
        self.cls = cls
        self.size = size
   
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        '''
        print('------image----')
        print(np.asarray(sample))
        print('------transform----')
        print(self.transform(sample))
        exit()
        '''
        if "good" in path:
            if isinstance(self.size, int):
                target = Image.new('L', (self.size, self.size))
            elif isinstance(self.size, list) and len(self.size) == 2:
                target = Image.new('L', tuple(self.size))
            else:
                raise ValueError(f'Invalid self.size: {self.size}')
            
            sample_class = 0
            is_mask = False
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            if os.path.exists(target_path):
                target = self.loader(target_path)
                is_mask = True
            else:
                target = None
                is_mask = False
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        
        if target is not None:
            target = target[:1]
        else:
            target = []

        return sample, target, sample_class, is_mask,path 

class StreamingDataset:
    """This dataset is made specifically for the streamlit app."""
    def __init__(self, size: int = 224):
        self.size = size
        self.transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        self.samples = []
    
    def add_pil_image(self, image : Image):
        image = image.convert('RGB')
        self.samples.append(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (self.transform(sample), tensor(0.))
