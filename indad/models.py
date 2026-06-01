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
from patchcore_normalization import (
    apply_position_normalization,
    compute_position_score_stats,
)
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


def save_tensor(results_dir, filename, x):
    path = os.path.join(results_dir, filename)
    m = Module()
    par = nn.Parameter(x)
    m.register_parameter("0", par)
    tensors = torch.jit.script(m)
    tensors.save(path)


def save_patchcore_archive(results_dir, filename, patch_lib, stats=None):
    path = os.path.join(results_dir, filename)
    m = Module()
    m.register_parameter("patch_lib", nn.Parameter(patch_lib.detach()))

    if stats is not None:
        m.register_parameter("score_baseline", nn.Parameter(stats["baseline"].detach()))
        m.register_parameter("score_scale", nn.Parameter(stats["scale"].detach()))
        threshold = stats["recommended_pixel_threshold"].detach().reshape(1)
        m.register_parameter("recommended_pixel_threshold", nn.Parameter(threshold))

    tensors = torch.jit.script(m)
    tensors.save(path)


def load_patchcore_archive(path):
    ts = torch.jit.load(path, map_location="cpu")
    params = {key: value.detach() for key, value in ts.named_parameters()}

    if "patch_lib" in params:
        patch_lib = params["patch_lib"]
    elif "0" in params:
        patch_lib = params["0"]
    else:
        raise ValueError(f"No patch library found in {path}")

    if "score_baseline" in params and "score_scale" in params:
        stats = {
            "baseline": params["score_baseline"],
            "scale": params["score_scale"],
            "recommended_pixel_threshold": params.get(
                "recommended_pixel_threshold",
                torch.tensor([0.0]),
            ).reshape(()),
        }
    else:
        stats = None

    patch_lib.requires_grad_(False)
    if stats is not None:
        stats["baseline"].requires_grad_(False)
        stats["scale"].requires_grad_(False)
        stats["recommended_pixel_threshold"].requires_grad_(False)

    return patch_lib, stats

def print_tensor(x,num):
    t = x.reshape(-1)
    for i in range(num):
        print(i,t[i].item())

def mask_to_polygon(mask, epsilon=2.0, close=True, threshold=127):
    """
    mask: HxW, uint8, 0/255 二值图
    epsilon: Douglas-Peucker 近似精度（像素）
    close:   是否返回首尾闭合的多边形 (True/False)
    return:  list[np.ndarray] 每个元素 shape (N,2) float32
    """
    # 1. 预处理
    mask = (mask > threshold).astype(np.uint8) * 255

    # 2. 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:          # 面积太小直接跳过
            continue
        # 3. DP 近似，提高阈值可以减少顶点数量
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon * peri * 0.01, True)
        pts = approx.squeeze()    # (N,2)
        if len(pts) < 3:
            continue
        if close and not np.all(pts[0] == pts[-1]):
            pts = np.vstack([pts, pts[0]])
        polygons.append(pts.astype(np.float32))
    return polygons

def polys_to_rects(polygons):
    """
    polygons: list[np.ndarray] 每个元素 shape (N,2) float32
    return:   list[np.ndarray] 每个元素 shape (4,2) float32
    """
    rects = []
    for pts in polygons:
        x, y, w, h = cv2.boundingRect(pts)
        rects.append(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32))
    return rects

def save_smap_image2(results_dir,img_path,sample,score,s_map,predict_time):
    threshold = 224
    
    #获取图片所在文件目录，文件名称
    filename=os.path.basename(img_path)
    filename=filename.split('.')[0]
    #获取图片分类名称
    classname=os.path.basename(os.path.dirname(img_path))
    #分数转字符串
    scorename = "{:.2f}".format( score.item() )
    predict_time_str = "{:.0f}".format( predict_time*1000 )
    #分割图路径
    smap_path = os.path.join(results_dir,classname + '_' + filename  + '_' +  scorename + '_' + predict_time_str + 'ms.jpg')
    

    img = tensor_to_img(sample[0], normalize=True)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # heatmap = np.transpose(heatmap, (1,2,0))
    # heatmap = s_map  / np.max(s_map)
    s_map = pred_to_img(s_map).cpu().numpy().squeeze()
    heatmap = (s_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(heatmap_color, 0.5, img, 0.5, 0)
    # cv2.imwrite(smap_path, result)

    polys = mask_to_polygon(heatmap, epsilon=1.5, threshold=threshold)
    rects = polys_to_rects(polys)
    
    # 画多边形和矩形
    for p in polys:
        cv2.polylines(result, [p.astype(int)], isClosed=True, color=(0,0,255), thickness=2)
    for rect in rects:
        cv2.polylines(result, [rect.astype(int)], isClosed=True, color=(0,255,0), thickness=2)
    
    stackimg = cv2.vconcat([img,result])

    cv2.imwrite(smap_path, stackimg)

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
        
        # return -1,-1

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
		jobini = None,
        match_mode: str = "exact_position" # 推理匹配方式: global | same_row | exact_position，训练库统一按 exact_position 保存
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
        self.match_mode = match_mode
        self.score_normalization_enabled = score_normalization_enabled
        self.score_normalization_min_train_patches = score_normalization_min_train_patches
        self.score_normalization_scale_floor_quantile = score_normalization_scale_floor_quantile
        self.score_normalization_scale_cap_quantile = score_normalization_scale_cap_quantile
        self.score_normalization_smooth_scale = score_normalization_smooth_scale
        self.score_normalization_smooth_kernel = score_normalization_smooth_kernel
        self.score_normalization_threshold_quantile = score_normalization_threshold_quantile
        self.score_normalization_clamp_min_zero = score_normalization_clamp_min_zero
        self.score_stats = None

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
            # width = self.largest_fmap_size[1]
            # patch = patch[:,:,:,int(0.1*width):int(0.9*width)] #去掉左右10%的patch
            if self.start_pos != 0 or self.end_pos !=0:
                patch = patch[:, :, :, int(self.start_pos / 8) : int(self.end_pos / 8)]
            
            patch = patch.permute(0,2,3,1)
            patch = patch.reshape(patch.shape[1],patch.shape[2], -1, patch.shape[-1]) # [H, W, 1, 384]
            self.patch_lib.append(patch)

            if self.jobini is not None:
                self.jobini.set_exec_progress(10 + (idx / len_ds) * 20)  # ?
        
        print('patch shape:',patch.shape)
        self.patch_lib = torch.cat(self.patch_lib, 2) # [H, W, len_ds, 384]

        print('patch_lib shape:',self.patch_lib.shape)

        # progress: 30 -> 95
        if self.f_coreset < 1:
            H = self.patch_lib.shape[0]
            W = self.patch_lib.shape[1]
            if self.max_feature_count == 0:
                n = min(int(self.f_coreset * self.patch_lib.shape[2]), 60000//(H*W))
            else:
                n = min(int(self.f_coreset * self.patch_lib.shape[2]), self.max_feature_count//(H*W))
            n = max(n, 1) # at least 1 patch per position, otherwise SparseRandomProjection will complain.
            # 输入大小:[H, W, N, C] 对每个位置进行采样子集
            sampled_positions = []
            for h in range(H):
                for w in range(W):
                    pos_lib = self.patch_lib[h, w]  # [N_train, C]
                    # 位置内 patch 数 <= n 时跳过 coreset，避免 SparseRandomProjection 出错
                    if pos_lib.shape[0] <= n:
                        sampled_positions.append(pos_lib)
                    else:
                        idx = get_coreset_idx_randomp(
                            pos_lib,
                            n,
                            eps=self.coreset_eps,
                            jobini=None,    # 子任务进度不上报，避免反复刷 ini
                        )
                        sampled_positions.append(pos_lib[idx])

                    # 上报每个位置的总进度 30 -> 95
                    if self.jobini is not None:
                        total_positions = H * W
                        current_position = h * W + w + 1
                        self.jobini.set_exec_progress(30 + (current_position / total_positions) * 65)
            self.patch_lib = torch.stack(sampled_positions, dim=0).reshape(H, W, -1, self.patch_lib.shape[-1])
            print('coreset size:',self.patch_lib.shape)

        self.score_stats = None
        if self.score_normalization_enabled:
            try:
                self.score_stats = compute_position_score_stats(
                    self.patch_lib,
                    min_train_patches=self.score_normalization_min_train_patches,
                    scale_floor_quantile=self.score_normalization_scale_floor_quantile,
                    scale_cap_quantile=self.score_normalization_scale_cap_quantile,
                    smooth_scale=self.score_normalization_smooth_scale,
                    smooth_kernel=self.score_normalization_smooth_kernel,
                    threshold_quantile=self.score_normalization_threshold_quantile,
                )
                print(
                    "score normalization threshold:",
                    self.score_stats["recommended_pixel_threshold"].item(),
                )
            except ValueError as exc:
                print(f"score normalization disabled: {exc}")
                self.score_stats = None

        save_patchcore_archive(self.results_dir, 'patch_lib.ts', self.patch_lib, self.score_stats)
        
    def load(self, path: str,fmap_size: list):
        self.patch_lib, self.score_stats = load_patchcore_archive(path)
        object.__setattr__(self, "resize", torch.nn.AdaptiveAvgPool2d(fmap_size))
        if self.score_stats is None:
            print("score normalization stats not found; using raw PatchCore scores")
        return True

    def _normalize_score_map_if_available(self, raw_map: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "score_normalization_enabled", True):
            return raw_map
        if self.match_mode != "exact_position":
            return raw_map
        if self.score_stats is None:
            return raw_map

        baseline = self.score_stats["baseline"].to(raw_map.device)
        scale = self.score_stats["scale"].to(raw_map.device)
        if baseline.shape != raw_map.shape or scale.shape != raw_map.shape:
            print(
                "score normalization stats shape mismatch; using raw PatchCore scores "
                f"raw={tuple(raw_map.shape)} baseline={tuple(baseline.shape)} scale={tuple(scale.shape)}"
            )
            return raw_map

        return apply_position_normalization(
            raw_map,
            baseline,
            scale,
            clamp_min_zero=getattr(self, "score_normalization_clamp_min_zero", True),
        )

    def predict(self, sample, path: str, neighbor_radius: int = 1):
        start_time = timeit.default_timer()
        feature_maps = self(sample) #sample [1,3,256,256] feature_maps [(1, C1, 32, 32), (1, C2, 16, 16)]
        if self.image_shape is None:
            self.image_shape = sample.shape
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]

        patch = torch.cat(resized_maps, 1) #patch [1, 384, 32, 32] [1,C1+C2, H, W]
        
        if self.start_pos != 0 or self.end_pos !=0: # start_pos / end_pos 裁剪
            patch = patch[:, :, :, int(self.start_pos / 8) : int(self.end_pos / 8)]
        
        fmap_h, fmap_w = patch.shape[-2], patch.shape[-1]  # 记录特征图空间尺寸（裁剪后）
        
        patch = patch.to(device)
        patch_lib = self.patch_lib.to(device)

        if self.match_mode == "global":
            # patch = patch.reshape(patch.shape[1], -1).T
            patch = patch.permute(0, 2, 3, 1).reshape(-1, patch.shape[1]) # [H*W, C]
            global_lib = patch_lib.reshape(-1, patch_lib.shape[-1]) # [H*W*N, C]
            dist = torch.cdist(patch, global_lib)   # [H*W, H*W*N]
            min_val, min_idx = torch.min(dist, dim=1)

        elif self.match_mode == "same_row":
            patch = patch.permute(0, 2, 3, 1).squeeze(0)   # [H, W, C]
            # patch: [H, W, C], row_lib: [H, W*N_pos, C]
            H, W, N_pos, C = patch_lib.shape
            row_lib = patch_lib.reshape(H, W * N_pos, C)
            r = neighbor_radius
            if r == 0:
                dist = torch.cdist(patch, row_lib) # [H, W, W*N_pos]
                min_val, min_idx = torch.min(dist, dim=2) # [H, W]
                min_val = min_val.reshape(-1)       # [H*W]
            else:
                lib_pad = torch.nn.functional.pad(
                    row_lib.permute(1, 2, 0),  # [W*N_pos, C, H]
                    (r, r), mode='replicate'
                ).permute(2, 0, 1)                       # [H+2r, W*N_pos, C]
                # 每行 h 取邻域 [h-r, h+r]，合并成 (2r+1)*W*N_pos 个候选
                # 用 unfold: lib_pad[h:h+2r+1] 是该行的候选
                # 一次性构造 [H, (2r+1)*W*N_pos, C]
                K = 2*r+1
                lib_pad = lib_pad.unfold(0, K, 1) # [H+2r, W*N_pos, C] -> [H, W*N_pos, C, K]
                lib_pad = lib_pad.permute(0, 3, 1, 2).contiguous() # [H, K, W*N_pos, C]
                lib_pad = lib_pad.reshape(H, K * W * N_pos, C)     # [H, K*W*N_pos, C]

                # patch: [H, W, C], lib_pad: [H, K*W*N_pos, C]
                print('patch shape:', patch.shape, 'patch_lib shape:', lib_pad.shape)
                dist = torch.cdist(patch, lib_pad) # [H, W, K*W*N_pos]
                min_val, min_idx = torch.min(dist, dim=2) # [H, W]
                min_val = min_val.reshape(-1)       # [H*W]

        elif self.match_mode == "exact_position":
            patch = patch.permute(0, 2, 3, 1).squeeze(0).unsqueeze(2)   # [H, W, 1, C]
            # patch: [H, W, 1, C], patch_lib: [H, W, N_pos, C]
            H, W, N_pos, C = patch_lib.shape
            r = neighbor_radius

            if r == 0:
                dist = torch.cdist(patch, patch_lib) # [H, W, 1, N_pos]
                min_val, min_idx = torch.min(dist, dim=-1) # [H, W, 1]
                min_val = min_val.reshape(-1)       # [H*W]
            else:
                lib_pad = torch.nn.functional.pad(
                    patch_lib.permute(2, 3, 0, 1),  # [N_pos, C, H, W]
                    (r, r, r, r), mode='replicate'
                ).permute(2, 3, 0, 1)                       # [H+2r, W+2r, N_pos, C]
                # 每位置 (h,w) 取邻域 [(h-r,h+r),(w-r,w+r)]，合并成 (2r+1)*(2r+1)*N_pos 个候选
                # 用 unfold: lib_pad[h:h+2r+1,w:w+2r+1] 是该位置的候选
                # 一次性构造 [H, W, (2r+1)*(2r+1)*N_pos, C]
                K = 2*r+1
                lib_pad = lib_pad.unfold(0, K, 1).unfold(1, K, 1) # [H+2r,W+2r,N_pos,C] -> [H,W,N_pos,C,K,K]
                lib_pad = lib_pad.permute(0, 1, 4, 5, 2, 3).contiguous() # [H,W,K,K,N_pos,C]
                lib_pad = lib_pad.reshape(H, W, K*K*N_pos, C)     # [H,W,K*K*N_pos,C]

                # patch: [H,W,1,C], lib_pad: [H,W,K*K*N_pos,C]
                print('patch shape:', patch.shape, 'patch_lib shape:', lib_pad.shape)
                dist = torch.cdist(patch, lib_pad) # [H,W,K*K*N_pos]
                min_val, min_idx = torch.min(dist, dim=-1) # [H,W]
                min_val = min_val.reshape(-1)       # [H*W]

        raw_map = min_val.view(fmap_h, fmap_w)
        score_map = self._normalize_score_map_if_available(raw_map)

        s_star = torch.max(score_map)
        s = s_star.cpu()

        if isinstance(self.smap_size, int):
            width, height = self.smap_size, self.smap_size
        elif isinstance(self.smap_size, list) and len(self.smap_size) == 2:
            width, height = self.smap_size[0], self.smap_size[1]
        else:
            raise ValueError(f'Invalid smap_size: {self.smap_size}')

        if self.start_pos != 0 or self.end_pos !=0: # start_pos / end_pos 裁剪
            width = int(self.end_pos) - int(self.start_pos)
            sample = sample[:, :, :, int(self.start_pos) : int(self.end_pos)]

        # # segmentation map
        s_map = score_map.view(1, 1, fmap_h, fmap_w)
        s_map = torch.nn.functional.interpolate(
            s_map, size=(height, width), mode='bilinear'
        )
        s_map = s_map.cpu()
        s_map = self.blur(s_map)

        # save smap image
        end_time = timeit.default_timer()
        # save_smap_image(self.results_dir, path, s, s_map, end_time - start_time)
        save_smap_image2(self.results_dir, path, sample, s, s_map, end_time - start_time)
        return s, s_map


    def get_parameters(self):
        return super().get_parameters({
            "f_coreset": self.f_coreset,
            "n_reweight": self.n_reweight,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "match_mode": self.match_mode,
            "score_normalization_enabled": self.score_normalization_enabled,
            "score_normalization_min_train_patches": self.score_normalization_min_train_patches,
            "score_normalization_scale_floor_quantile": self.score_normalization_scale_floor_quantile,
            "score_normalization_scale_cap_quantile": self.score_normalization_scale_cap_quantile,
            "score_normalization_smooth_scale": self.score_normalization_smooth_scale,
            "score_normalization_smooth_kernel": self.score_normalization_smooth_kernel,
            "score_normalization_threshold_quantile": self.score_normalization_threshold_quantile,
            "score_normalization_clamp_min_zero": self.score_normalization_clamp_min_zero,
            "score_normalization_has_stats": self.score_stats is not None,
        })
