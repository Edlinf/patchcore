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
from sklearn.metrics import roc_auc_score

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
    direct = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if direct:
        return direct
    images = []
    for child in sorted(path.iterdir()):
        if child.is_dir():
            images.extend(sorted(p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    return images


def infer_label_from_path(path):
    parent = Path(path).parent.name.lower()
    if parent in ("good", "ok", "normal"):
        return 0
    if parent in ("bad", "ng", "defect", "abnormal"):
        return 1
    return -1


def load_patchcore_archive_simple(path):
    # 训练归档里主要保存 patch_lib；新版本还会保存每位置的 score baseline/scale。
    # patch_lib 形状约定为 [H, W, N, C]：特征图高、宽、每位置样本数、特征维度。
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
    # raw_map / baseline / scale 都是 [H, W]。
    # 仅在 exact_position 模式下使用，用每个位置自己的正常分数分布做归一化。
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


def _patch_to_hwc(patch):
    # patch: [1, C, H, W] -> [H, W, C]
    return patch.permute(0, 2, 3, 1).squeeze(0)


def raw_map_global(patch, patch_lib):
    # patch: [1, C, H, W]
    # patch_lib: [H, W, N, C]
    # global 模式把所有位置展开成一个大库：[H*W*N, C]。
    patch_hwc = _patch_to_hwc(patch)
    H, W, C = patch_hwc.shape
    query = patch_hwc.reshape(-1, C)  # [H*W, C]
    lib = patch_lib.reshape(-1, patch_lib.shape[-1])  # [H*W*N, C]
    dist = torch.cdist(query, lib)  # [H*W, H*W*N]
    return torch.min(dist, dim=1).values.reshape(H, W)  # [H, W]


def raw_map_same_row(patch, patch_lib, neighbor_radius=0):
    # patch: [1, C, H, W] -> patch_hwc: [H, W, C]
    # patch_lib: [H, W, N, C]
    # same_row 模式：第 h 行的每个 query patch 只和同一行或邻近行的 memory bank 匹配。
    patch_hwc = _patch_to_hwc(patch)
    H, W, N, C = patch_lib.shape
    row_lib = patch_lib.reshape(H, W * N, C)  # [H, W*N, C]
    r = int(neighbor_radius)
    if r == 0:
        # patch_hwc: [H, W, C], row_lib: [H, W*N, C]
        # cdist 按 batch 维 H 分行计算，输出 [H, W, W*N]。
        dist = torch.cdist(patch_hwc, row_lib)
        return torch.min(dist, dim=2).values  # [H, W]

    K = 2 * r + 1
    # row_lib.permute(1, 2, 0): [W*N, C, H]
    # pad 行方向后恢复为 [H+2r, W*N, C]。
    lib_pad = torch.nn.functional.pad(
        row_lib.permute(1, 2, 0),
        (r, r),
        mode="replicate",
    ).permute(2, 0, 1)
    # unfold 行邻域：[H, W*N, C, K] -> [H, K, W*N, C]
    lib_pad = lib_pad.unfold(0, K, 1)
    lib_pad = lib_pad.permute(0, 3, 1, 2).contiguous()
    lib_pad = lib_pad.reshape(H, K * W * N, C)  # [H, K*W*N, C]
    dist = torch.cdist(patch_hwc, lib_pad)  # [H, W, K*W*N]
    return torch.min(dist, dim=2).values  # [H, W]


def raw_map_exact_position(patch, patch_lib, neighbor_radius=0):
    # patch: [1, C, H, W] -> patch_hwc: [H, W, C]
    # patch_lib: [H, W, N, C]
    # exact_position 模式：每个位置只和同位置 memory bank 匹配；r>0 时加入周围 K*K 个位置。
    patch_hwc = _patch_to_hwc(patch)
    H, W, N, C = patch_lib.shape
    r = int(neighbor_radius)
    patch_query = patch_hwc.unsqueeze(2)  # [H, W, 1, C]
    if r == 0:
        # patch_query: [H, W, 1, C], patch_lib: [H, W, N, C]
        # 输出 [H, W, 1, N]，对 N 取最小后得到 [H, W]。
        dist = torch.cdist(patch_query, patch_lib)
        return torch.min(dist, dim=-1).values.reshape(H, W)

    K = 2 * r + 1
    # patch_lib.permute(2, 3, 0, 1): [N, C, H, W]
    # pad 空间维后恢复为 [H+2r, W+2r, N, C]。
    lib_pad = torch.nn.functional.pad(
        patch_lib.permute(2, 3, 0, 1),
        (r, r, r, r),
        mode="replicate",
    ).permute(2, 3, 0, 1)
    # unfold 取每个位置周围 K*K 邻域：
    # [H, W, N, C, K, K] -> [H, W, K, K, N, C] -> [H, W, K*K*N, C]
    lib_pad = lib_pad.unfold(0, K, 1).unfold(1, K, 1)
    lib_pad = lib_pad.permute(0, 1, 4, 5, 2, 3).contiguous()
    lib_pad = lib_pad.reshape(H, W, K * K * N, C)
    dist = torch.cdist(patch_query, lib_pad)  # [H, W, 1, K*K*N]
    return torch.min(dist, dim=-1).values.reshape(H, W)


def select_score_map(raw_map, stats, match_mode):
    if match_mode == "exact_position":
        return apply_score_stats(raw_map, stats)
    return raw_map


def raw_map_by_mode(patch, patch_lib, match_mode, neighbor_radius):
    if match_mode == "global":
        return raw_map_global(patch, patch_lib)
    if match_mode == "same_row":
        return raw_map_same_row(patch, patch_lib, neighbor_radius=neighbor_radius)
    if match_mode == "exact_position":
        return raw_map_exact_position(patch, patch_lib, neighbor_radius=neighbor_radius)
    raise ValueError(f"unsupported match_mode: {match_mode}")


class PatchCorePredictor:
    def __init__(self, model_path, backbone="resnet18", out_indices=(2, 3), image_size=(224, 224), fmap_size=None, resize_method="cv2", match_mode="exact_position", neighbor_radius=0, output_dir="./results-predict-simple"):
        self.model_path = Path(model_path)
        self.backbone = backbone
        self.out_indices = tuple(out_indices)
        self.image_size = list(image_size)
        self.fmap_size = list(fmap_size) if fmap_size is not None else None
        self.resize_method = resize_method
        self.match_mode = match_mode
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
        # sample: [1, 3, image_h, image_w]
        with torch.no_grad():
            feature_maps = self.feature_extractor(sample.to(self.device))
        if self.resize is None:
            self.fmap_size = list(feature_maps[0].shape[-2:])
            self.resize = torch.nn.AdaptiveAvgPool2d(self.fmap_size)
        # feature_maps: 多层特征 [1, C_i, H_i, W_i]
        # average 后 resize 到同一个 [H, W]，再按通道拼接成 patch: [1, C_total, H, W]。
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        return torch.cat(resized_maps, 1)

    def predict_tensor(self, sample):
        patch = self.extract_patch(sample)  # [1, C, H, W]
        raw_map = raw_map_by_mode(
            patch,
            self.patch_lib,
            match_mode=self.match_mode,
            neighbor_radius=self.neighbor_radius,
        )
        # raw_map / score_map: [H, W]，score 是整张图的最大异常分数。
        score_map = select_score_map(raw_map, self.score_stats, self.match_mode)
        score = torch.max(score_map).detach().cpu()
        return score, score_map

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        sample = self.transform(image).unsqueeze(0)
        return image, sample

    def predict_image(self, image_path):
        start = time.time()
        image, sample = self.preprocess_image(image_path)
        score, score_map = self.predict_tensor(sample)
        elapsed_ms = (time.time() - start) * 1000
        return image, float(score.item()), score_map, elapsed_ms


def save_heatmap_outputs(image, score_map, image_path, output_dir, label, score, elapsed_ms):
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    classname = Path(image_path).parent.name
    score_text = f"{score:.2f}"
    elapsed_text = f"{elapsed_ms:.0f}ms"
    out_name = f"{classname}_{stem}_{score_text}_{elapsed_text}.jpg"

    # score_map: [H, W]，先归一化成 0-255，再 resize 回原图尺寸做可视化。
    heat = normalize_vis_map(score_map)
    heat = cv2.resize(heat, image.size)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat_color, 0.5, image_bgr, 0.5, 0)
    combined = cv2.hconcat([heat_color, overlay])
    combined_path = heatmap_dir / out_name
    cv2.imwrite(str(combined_path), combined)
    return combined_path


def write_scores_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "elapsed_ms", "result_path"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(path, rows):
    labels = [int(r["label"]) for r in rows if int(r["label"]) >= 0]
    scores = [float(r["score"]) for r in rows if int(r["label"]) >= 0]
    label_set = set(labels)
    if len(label_set) > 1:
        image_rocauc = float(roc_auc_score(labels, scores))
    else:
        image_rocauc = -1
    path.write_text(
        "{\n"
        f"  \"image_rocauc\": {image_rocauc},\n"
        f"  \"num_images\": {len(rows)},\n"
        f"  \"num_labeled\": {len(labels)},\n"
        f"  \"num_positive\": {labels.count(1)},\n"
        f"  \"num_negative\": {labels.count(0)}\n"
        "}\n",
        encoding="utf-8",
    )
    return image_rocauc


@click.command()
@click.option("--model", "model_path", required=True, type=Path)
@click.option("--image", type=Path, default=None)
@click.option("--input", "input_path", type=Path, default=None)
@click.option("--output", "output_dir", type=Path, default=Path("./results-predict-simple"))
@click.option("--backbone", default="")
@click.option("--image-size", default="")
@click.option("--fmap-size", default="")
@click.option("--resize-method", default="")
@click.option("--out-indices", default="2,3")
@click.option("--match-mode", default="exact_position", type=click.Choice(["global", "same_row", "exact_position"]))
@click.option("--neighbor-radius", default=0, type=int)
def cli_interface(model_path, image, input_path, output_dir, backbone, image_size, fmap_size, resize_method, out_indices, match_mode, neighbor_radius):
    if image is None and input_path is None:
        raise click.UsageError("Provide --image or --input")
    try:
        info = parse_model_info_simple(model_path)
    except ValueError:
        info = {}
    backbone = backbone or info.get("backbone", "resnet18")
    resize_method = resize_method or info.get("resize_method", "cv2")
    image_size = parse_pair(image_size) if image_size else info.get("image_size", [224, 224])
    fmap_size = parse_pair(fmap_size) if fmap_size else info.get("fmap_size")
    out_indices = tuple(int(i) for i in out_indices.split(","))

    predictor = PatchCorePredictor(
        model_path=model_path,
        backbone=backbone,
        out_indices=out_indices,
        image_size=image_size,
        fmap_size=fmap_size,
        resize_method=resize_method,
        match_mode=match_mode,
        neighbor_radius=neighbor_radius,
        output_dir=output_dir,
    ).load()

    images = [image] if image is not None else collect_images(input_path)
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        src_image, score, score_map, elapsed_ms = predictor.predict_image(image_path)
        label = infer_label_from_path(image_path)
        result_path = save_heatmap_outputs(src_image, score_map, image_path, output_dir, label, score, elapsed_ms)
        rows.append({
            "path": str(image_path),
            "label": label,
            "score": score,
            "elapsed_ms": round(elapsed_ms, 2),
            "result_path": str(result_path),
        })
        print(f"{image_path}: label={label}, score={score:.4f}, elapsed_ms={elapsed_ms:.1f}")
    write_scores_csv(output_dir / "scores.csv", rows)
    image_rocauc = write_metrics_json(output_dir / "metrics.json", rows)
    print(f"image_rocauc={image_rocauc:.4f}")


if __name__ == "__main__":
    cli_interface()
