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


if __name__ == "__main__":
    pass
