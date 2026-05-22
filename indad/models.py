from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models,transforms
import timm
import timeit
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
from PIL import Image
import os
from data import IMAGENET_MEAN, IMAGENET_STD

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x = x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x):
    range_min = x.min()
    range_max = x.max()
    y = x - range_min
    if (range_max - range_min) > 0:
        y /= (range_max - range_min)
    #return tensor_to_img(x)
    return y;

class Module(nn.Module):
    pass
    
def save_tensor(results_dir,filename,x):
    path = os.path.join(results_dir,filename)
    m = Module()
    par = nn.Parameter(x)
    m.register_parameter("0",par)
    tensors = torch.jit.script(m)
    tensors.save(path)

def print_tensor(x,num):
    t = x.reshape(-1)
    for i in range(num):
        print(i,t[i].item())

def save_smap_image(results_dir,img_path,score,s_map,predict_time):
    #获取图片所在文件目录，文件名称
    filename=os.path.basename(img_path)
    filename=filename.split('.')[0]
    #获取图片分类名称
    classname=os.path.basename(os.path.dirname(img_path))
    #分数转字符串
    scorename = "{:.2f}".format( score.item() )
    predict_time_str = "{:.0f}".format( predict_time*1000 )
    #分割图路径
    #smap_path = os.path.join(results_dir,classname + '_' + filename  + '_' +  scorename + '_' + predict_time_str + 'ms.jpg')
    smap_path = os.path.join(results_dir,classname + '_' +  scorename + '_' + filename   + '_' + predict_time_str + 'ms.jpg')

    tf = transforms.ToPILImage()
    img = tf(pred_to_img(s_map))
    img.save(smap_path)
    
class KNNExtractor(torch.nn.Module):
    def __init__(
        self,
        backbone_name : str = "resnet50",
        out_indices : Tuple = None,
        pool_last : bool = False,
        results_dir : str = './results',
    ):
        super().__init__()
        self.results_dir = results_dir        
        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=True,
            pretrained=True,
        )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name # for results metadata
        self.out_indices = out_indices

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = self.feature_extractor.to(self.device)
            
    def __call__(self, x: tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool:
            # spit into fmaps and z
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
        
    def predict(self, _: tensor, path: str):
        raise NotImplementedError

    def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
        """Calls predict step for each test sample."""
        '''
        image_preds = []
        image_labels = []
        image_labels_set = set()
        pixel_preds = []
        pixel_labels = []
        pixel_labels_set = set()
        for sample, mask, label, is_mask,path in tqdm(test_dl, **get_tqdm_params()):
            z_score, fmap = self.predict(sample,path[0])
            
            image_preds.append(z_score.numpy())
            image_labels.append(label.item())
            image_labels_set.add(label.item())
            #print('evaluate:',path,label.item(),z_score.numpy());
            
            if label.item() == 0 or is_mask.item() == True:  #OK图或者NG有像素级遮罩才计算
                pixel_preds.extend(fmap.flatten().numpy())
                pixel_labels.extend(mask.flatten().numpy())
                pixel_labels_set.add(label.item())
            
        image_preds = np.stack(image_preds)
        
        if len(image_labels_set) > 1: #验证集超过一个类别才能计算image_rocauc
            image_rocauc = roc_auc_score(image_labels, image_preds)
        else:
            image_rocauc = -1
        
        if len(pixel_labels_set) > 1: #超过一个类别才能计算roc_auc_score
            pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)
        else:
            pixel_rocauc = -1

        return image_rocauc, pixel_rocauc
        '''
        return -1,-1

    def get_parameters(self, extra_params : dict = None) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            **extra_params,
        }

class SPADE(KNNExtractor):
    def __init__(
        self,
        k: int = 5,
        backbone_name: str = "resnet18",
        out_indices: tuple = (1,2,3,-1),
        results_dir : str = './results',
        image_size: int = 224,
        max_feature_count: int = 0,
        jobini = None
    ):
        super().__init__(
            backbone_name=backbone_name,
            out_indices=out_indices,
            pool_last=True,
            results_dir=results_dir,
            image_size=image_size,
        )
        self.k = k
        self.image_size = image_size
        self.z_lib = []
        self.feature_maps = []
        self.threshold_z = None
        self.threshold_fmaps = None
        self.blur = GaussianBlur(4)

    def fit(self, train_dl):
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps, z = self(sample)

            # z vector
            self.z_lib.append(z)

            # feature maps
            if len(self.feature_maps) == 0:
                for fmap in feature_maps:
                    self.feature_maps.append([fmap])
            else:
                for idx, fmap in enumerate(feature_maps):
                    self.feature_maps[idx].append(fmap)

        self.z_lib = torch.vstack(self.z_lib)
        
        for idx, fmap in enumerate(self.feature_maps):
            self.feature_maps[idx] = torch.vstack(fmap)

    def predict(self, sample, path: str):
        feature_maps, z = self(sample)

        distances = torch.linalg.norm(self.z_lib - z, dim=1)
        values, indices = torch.topk(distances.squeeze(), self.k, largest=False)

        z_score = values.mean()

        # Build the feature gallery out of the k nearest neighbours.
        # The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
        # Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
        scaled_s_map = torch.zeros(1,1,self.image_size,self.image_size)
        for idx, fmap in enumerate(feature_maps):
            nearest_fmaps = torch.index_select(self.feature_maps[idx], 0, indices)
            # min() because kappa=1 in the paper
            s_map, _ = torch.min(torch.linalg.norm(nearest_fmaps - fmap, dim=1), 0, keepdims=True)
            scaled_s_map += torch.nn.functional.interpolate(
                s_map.unsqueeze(0), size=(self.image_size,self.image_size), mode='bilinear'
            )

        scaled_s_map = self.blur(scaled_s_map)
        
        return z_score, scaled_s_map

    def get_parameters(self):
        return super().get_parameters({
            "k": self.k,
        })

class PaDiM(KNNExtractor):
    def __init__(
        self,
        d_reduced: int = 100,
        backbone_name: str = "resnet18",
        out_indices: tuple = (1,2,3),
        results_dir : str = './results',
        max_feature_count: int = 0,
        jobini = None
    ):
        super().__init__(
            backbone_name=backbone_name,
            out_indices=out_indices,
            results_dir=results_dir,            
            image_size=image_size,
        )
        self.image_size = image_size
        self.d_reduced = d_reduced # your RAM will thank you
        self.epsilon = 0.04 # cov regularization
        self.patch_lib = []
        self.resize = None

    def fit(self, train_dl):
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps = self(sample)
            if self.resize is None:
                largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
            resized_maps = [self.resize(fmap) for fmap in feature_maps]
            self.patch_lib.append(torch.cat(resized_maps, 1))
        self.patch_lib = torch.cat(self.patch_lib, 0)

        # random projection
        if self.patch_lib.shape[1] > self.d_reduced:
            print(f"   PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
            self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]
            self.patch_lib_reduced = self.patch_lib[:,self.r_indices,...]
        else:
            print("   PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
            self.patch_lib_reduced = self.patch_lib

        # calcs
        self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)
        self.means_reduced = self.means[:,self.r_indices,...]
        x_ = self.patch_lib_reduced - self.means_reduced

        # cov calc
        self.E = torch.einsum(
            'abkl,bckl->ackl',
            x_.permute([1,0,2,3]), # transpose first two dims
            x_,
        ) * 1/(self.patch_lib.shape[0]-1)
        self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
        self.E_inv = torch.linalg.inv(self.E.permute([2,3,0,1])).permute([2,3,0,1])

    def predict(self, sample, path: str):
        feature_maps = self(sample)
        resized_maps = [self.resize(fmap) for fmap in feature_maps]
        fmap = torch.cat(resized_maps, 1)

        # reduce
        x_ = fmap[:,self.r_indices,...] - self.means_reduced

        left = torch.einsum('abkl,bckl->ackl', x_, self.E_inv)
        s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))
        scaled_s_map = torch.nn.functional.interpolate(
            s_map.unsqueeze(0), size=(self.image_size,self.image_size), mode='bilinear'
        )
        return torch.max(s_map), scaled_s_map[0, ...]

    def get_parameters(self):
        return super().get_parameters({
            "d_reduced": self.d_reduced,
            "epsilon": self.epsilon,
        })

class PatchCore(KNNExtractor):
    def __init__(
        self,
        f_coreset: float = 0.01, # fraction the number of training samples
        backbone_name : str = "resnet18",
        out_indices: tuple = (2,3),
        coreset_eps: float = 0.90, # sparse projection parameter
        results_dir : str = './results',
        image_size: int = 224,
        max_feature_count: int = 0,
        start_pos: int = 0, 
        end_pos: int = 0,
		jobini = None
    ):
        super().__init__(
            backbone_name=backbone_name,
            out_indices=out_indices,  # default (2,3)
            results_dir=results_dir,
        )
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.smap_size = image_size
        self.max_feature_count = max_feature_count
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.jobini = jobini
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = GaussianBlur(4)
        self.n_reweight = 3

        self.patch_lib = []
        self.largest_fmap_size = None
        self.image_shape = None
        self.resize = None
    
    def get_fmap_size(self):
        return self.largest_fmap_size
        
    def get_image_shape(self):
        # channel,width,height
        shape = [self.image_shape[1],self.image_shape[2],self.image_shape[3]]
        return shape
        
    def fit(self, train_dl):
        # progress: 10 -> 30
        len_ds = len(train_dl)
        for idx, (sample, _) in enumerate(tqdm(train_dl, **get_tqdm_params())):
            feature_maps = self(sample)
            if self.image_shape is None:
                self.image_shape = sample.shape
            if self.resize is None:
                self.largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(self.largest_fmap_size)
            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)#patch [1, 384, 32, 32] [1,C1+C2, H, W]
            
            width = self.largest_fmap_size[1]
            # patch = patch[:,:,:,int(0.1*width):int(0.9*width)] #去掉左右10%的patch
            if self.start_pos != 0 or self.end_pos !=0:
                patch = patch[:, :, :, int(self.start_pos / 8) : int(self.end_pos / 8)]

            patch = patch.reshape(patch.shape[1], -1).T #[1024, 384]
            self.patch_lib.append(patch)
            
            if self.jobini is not None:
                self.jobini.set_exec_progress(10 + (idx / len_ds) * 20)  # ?

        self.patch_lib = torch.cat(self.patch_lib, 0)

        # progress: 30 -> 95
        if self.f_coreset < 1:
            if self.max_feature_count == 0:
                # n = int(self.f_coreset * self.patch_lib.shape[0])   # RESERVED
                n = min(int(self.f_coreset * self.patch_lib.shape[0]), 60000)
            else:
                n = min(int(self.f_coreset * self.patch_lib.shape[0]), self.max_feature_count)
                
            self.coreset_idx = get_coreset_idx_randomp(
                self.patch_lib,
                n,
                eps=self.coreset_eps,
                jobini=self.jobini
            )
            self.patch_lib = self.patch_lib[self.coreset_idx]
            save_tensor(self.results_dir,'patch_lib.ts',self.patch_lib)
        
    def load(self, path: str,fmap_size: list):
        ts = torch.load(path)
        par = ts.named_parameters()
        for key,value in par:
            self.patch_lib = value
            self.patch_lib.requires_grad_(False)
            break
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        return True

    def predict(self, sample, path: str):
        start_time = timeit.default_timer()
        feature_maps = self(sample) #sample [1,3,256,256] feature_maps [(1, C1, 32, 32), (1, C2, 16, 16)]
        if self.image_shape is None:
            self.image_shape = sample.shape
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]

        patch = torch.cat(resized_maps, 1) #patch [1, 384, 32, 32] [1,C1+C2, H, W]

        patch = patch.to(device)
        patch = patch.reshape(patch.shape[1], -1).T
        self.patch_lib = self.patch_lib.to(device)
        
        dist = torch.cdist(patch, self.patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        
        # reweighting
        m_test = patch[s_idx].unsqueeze(0) # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0) # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib) # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
        s = w*s_star
        s = s.cpu()

        if isinstance(self.smap_size, int):
            width, height = self.smap_size, self.smap_size
        elif isinstance(self.smap_size, list) and len(self.smap_size) == 2:
            width, height = self.smap_size[0], self.smap_size[1]
        else:
            raise ValueError(f'Invalid smap_size: {self.smap_size}')
            
        # # segmentation map
        s_map = min_val.view(1,1,*feature_maps[0].shape[-2:])
        """
        s_map = torch.nn.functional.interpolate(
            s_map, size=(self.smap_size,self.smap_size), mode='bilinear'
        )
        """
        s_map = torch.nn.functional.interpolate(
            s_map, size=(height, width), mode='bilinear'
        )
        s_map = s_map.cpu()
        s_map = self.blur(s_map)

        # save smap image
        end_time = timeit.default_timer()
        save_smap_image(self.results_dir, path, s, s_map, end_time - start_time)
        
        return s, s_map


    def get_parameters(self):
        return super().get_parameters({
            "f_coreset": self.f_coreset,
            "n_reweight": self.n_reweight,
        })
