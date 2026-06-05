import csv
import os
import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import Cv2AdaptiveResize, IMAGENET_MEAN, IMAGENET_STD, TransformAdaptiveResize

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_pair(value):
    if isinstance(value, (list, tuple)):
        return [int(value[0]), int(value[1])]
    sep = "," if "," in value else "x"
    parts = [p.strip() for p in str(value).split(sep)]
    if len(parts) != 2:
        raise ValueError(f"expected pair like 1280,128 or 48x16, got {value}")
    return [int(parts[0]), int(parts[1])]


def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_patchcore_archive_simple(path):
    ts = torch.jit.load(str(path), map_location="cpu")
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
            "recommended_pixel_threshold": params.get("recommended_pixel_threshold", torch.tensor([0.0])).reshape(()),
        }
    else:
        stats = None

    patch_lib.requires_grad_(False)
    if stats is not None:
        stats["baseline"].requires_grad_(False)
        stats["scale"].requires_grad_(False)
        stats["recommended_pixel_threshold"].requires_grad_(False)
    return patch_lib, stats


def parse_model_info_simple(model_path):
    stem = Path(model_path).name.split(".")[0]
    parts = stem.split("_")
    if len(parts) != 9:
        raise ValueError(f"Cannot parse model filename: {model_path}")
    method, jobno, resize_method, backbone, out_indices, fmap_size, image_shape, precision, md5 = parts
    if out_indices != "23":
        raise ValueError(f"Only out_indices=23 supported by simple predictor, got {out_indices}")
    image_shape = [int(i) for i in image_shape.split("x")]
    return {
        "method": method,
        "jobno": jobno,
        "resize_method": resize_method,
        "backbone": backbone.replace("-", "_"),
        "out_indices": [2, 3],
        "fmap_size": [int(i) for i in fmap_size.split("x")],
        "image_size": [image_shape[1], image_shape[2]],
    }


def apply_score_stats(raw_map, stats):
    if stats is None:
        return raw_map
    baseline = stats["baseline"].to(raw_map.device)
    scale = stats["scale"].to(raw_map.device)
    if baseline.shape != raw_map.shape or scale.shape != raw_map.shape:
        return raw_map
    return torch.clamp_min((raw_map.float() - baseline.float()) / scale.float(), 0.0)


def build_transform(image_size, resize_method):
    resize = Cv2AdaptiveResize(image_size) if resize_method == "cv2" else TransformAdaptiveResize(image_size)
    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def normalize_vis_map(score_map):
    values = score_map.detach().cpu().float()
    values = values - values.min()
    if values.max() > 0:
        values = values / values.max()
    return (values.numpy() * 255).astype(np.uint8)


class PatchCorePredictor:
    def __init__(self, model_path, backbone="resnet18", out_indices=(2, 3), image_size=(224, 224), fmap_size=None, resize_method="cv2", neighbor_radius=0, output_dir="./results-predict-simple"):
        self.model_path = Path(model_path)
        self.backbone = backbone
        self.out_indices = tuple(out_indices)
        self.image_size = list(image_size)
        self.fmap_size = list(fmap_size) if fmap_size is not None else None
        self.resize_method = resize_method
        self.neighbor_radius = int(neighbor_radius)
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = None
        self.feature_extractor = None
        self.patch_lib = None
        self.score_stats = None
        self.transform = build_transform(self.image_size, self.resize_method)

    def load(self):
        project_root = Path(__file__).resolve().parents[1]
        torch.hub.set_dir(str(project_root / "hub"))
        self.feature_extractor = timm.create_model(
            self.backbone,
            out_indices=self.out_indices,
            features_only=True,
            pretrained=True,
        )
        self.feature_extractor.eval().to(self.device)
        self.patch_lib, self.score_stats = load_patchcore_archive_simple(self.model_path)
        self.patch_lib = self.patch_lib.to(self.device)
        if self.fmap_size is not None:
            self.resize = torch.nn.AdaptiveAvgPool2d(self.fmap_size)
        return self

    def extract_patch(self, sample):
        with torch.no_grad():
            feature_maps = self.feature_extractor(sample.to(self.device))
        if self.resize is None:
            self.fmap_size = list(feature_maps[0].shape[-2:])
            self.resize = torch.nn.AdaptiveAvgPool2d(self.fmap_size)
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        return torch.cat(resized_maps, 1)


if __name__ == "__main__":
    pass
