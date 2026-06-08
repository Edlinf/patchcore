import csv
import os
import sys
import time
from dataclasses import dataclass
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


@dataclass
class BigTile:
    index: int
    row: int
    col: int
    image: Image.Image
    box: tuple


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
        # image_shape is saved as [C, H, W], while Cv2AdaptiveResize expects [W, H].
        "image_size": [image_shape[2], image_shape[1]],
    }


def apply_score_stats(raw_map, stats):
    # raw_map 可以是 [H, W] 或 [B, H, W]；baseline / scale 是 [H, W]。
    # 仅在 exact_position 模式下使用，用每个位置自己的正常分数分布做归一化。
    if stats is None:
        return raw_map
    baseline = stats["baseline"].to(raw_map.device)
    scale = stats["scale"].to(raw_map.device)
    if raw_map.shape[-2:] != baseline.shape or raw_map.shape[-2:] != scale.shape:
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


def split_big_image_by_geometry(image, rows, cols, top_margin, bottom_margin, left_margin, right_margin, hori_gap, vert_gap):
    width, height = image.size
    available_width = width - left_margin - right_margin - (cols - 1) * hori_gap
    available_height = height - top_margin - bottom_margin - (rows - 1) * vert_gap
    if available_width <= 0 or available_height <= 0:
        raise ValueError("invalid big-image geometry parameters")
    tile_w = available_width // cols
    tile_h = available_height // rows

    tiles = []
    index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = left_margin + col * (tile_w + hori_gap)
            y1 = top_margin + row * (tile_h + vert_gap)
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            box = (x1, y1, x2, y2)
            tiles.append(BigTile(index=index, row=row, col=col, image=image.crop(box), box=box))
            index += 1
    return tiles


def _patch_to_hwbc(patch):
    # patch: [B, C, H, W] -> [H, W, B, C]
    return patch.permute(2, 3, 0, 1)


def raw_map_global(patch, patch_lib):
    # patch: [B, C, H, W]
    # patch_lib: [H, W, N, C]
    # global 模式不关心空间位置，query 直接展开成 [B*H*W, C]。
    B, C, H, W = patch.shape
    query = patch.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
    lib = patch_lib.reshape(-1, patch_lib.shape[-1])  # [H*W*N, C]
    dist = torch.cdist(query, lib)  # [B*H*W, H*W*N]
    return torch.min(dist, dim=1).values.reshape(B, H, W)  # [B, H, W]


def raw_map_same_row(patch, patch_lib, neighbor_radius=0):
    # patch: [B, C, H, W] -> patch_hwbc: [H, W, B, C]
    # same_row 模式先把每一行的 W 个位置和 B 张图合并成 query: [H, W*B, C]。
    patch_hwbc = _patch_to_hwbc(patch)
    H, W, N, C = patch_lib.shape
    B = patch_hwbc.shape[2]
    query = patch_hwbc.reshape(H, W * B, C)  # [H, W*B, C]
    row_lib = patch_lib.reshape(H, W * N, C)  # [H, W*N, C]
    r = int(neighbor_radius)
    if r == 0:
        dist = torch.cdist(query, row_lib)  # [H, W*B, W*N]
        raw_h_wb = torch.min(dist, dim=2).values  # [H, W*B]
        return raw_h_wb.reshape(H, W, B).permute(2, 0, 1)  # [B, H, W]

    K = 2 * r + 1
    lib_pad = torch.nn.functional.pad(
        row_lib.permute(1, 2, 0),
        (r, r),
        mode="replicate",
    ).permute(2, 0, 1)  # [H+2r, W*N, C]
    lib_pad = lib_pad.unfold(0, K, 1)
    lib_pad = lib_pad.permute(0, 3, 1, 2).contiguous()
    lib_pad = lib_pad.reshape(H, K * W * N, C)  # [H, K*W*N, C]
    dist = torch.cdist(query, lib_pad)  # [H, W*B, K*W*N]
    raw_h_wb = torch.min(dist, dim=2).values  # [H, W*B]
    return raw_h_wb.reshape(H, W, B).permute(2, 0, 1)  # [B, H, W]


def raw_map_exact_position(patch, patch_lib, neighbor_radius=0):
    # patch: [B, C, H, W] -> patch_hwbc: [H, W, B, C]
    # patch_lib: [H, W, N, C]
    # 先得到 raw_hwb: [H, W, B]，最后转成 raw_bhw: [B, H, W]。
    patch_hwbc = _patch_to_hwbc(patch)
    H, W, N, C = patch_lib.shape
    r = int(neighbor_radius)
    if r == 0:
        dist = torch.cdist(patch_hwbc, patch_lib)  # [H, W, B, N]
        raw_hwb = torch.min(dist, dim=-1).values  # [H, W, B]
        return raw_hwb.permute(2, 0, 1)  # [B, H, W]

    K = 2 * r + 1
    lib_pad = torch.nn.functional.pad(
        patch_lib.permute(2, 3, 0, 1),
        (r, r, r, r),
        mode="replicate",
    ).permute(2, 3, 0, 1)  # [H+2r, W+2r, N, C]
    lib_pad = lib_pad.unfold(0, K, 1).unfold(1, K, 1)
    lib_pad = lib_pad.permute(0, 1, 4, 5, 2, 3).contiguous()
    lib_pad = lib_pad.reshape(H, W, K * K * N, C)  # [H, W, K*K*N, C]
    dist = torch.cdist(patch_hwbc, lib_pad)  # [H, W, B, K*K*N]
    raw_hwb = torch.min(dist, dim=-1).values  # [H, W, B]
    return raw_hwb.permute(2, 0, 1)  # [B, H, W]


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
        # sample: [B, 3, image_h, image_w]
        with torch.no_grad():
            feature_maps = self.feature_extractor(sample.to(self.device))
        if self.resize is None:
            self.fmap_size = list(feature_maps[0].shape[-2:])
            self.resize = torch.nn.AdaptiveAvgPool2d(self.fmap_size)
        # feature_maps: 多层特征 [B, C_i, H_i, W_i]
        # average 后 resize 到同一个 [H, W]，再按通道拼接成 patch: [B, C_total, H, W]。
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        return torch.cat(resized_maps, 1)

    def predict_batch_tensor(self, batch):
        patch = self.extract_patch(batch)  # [B, C, H, W]
        raw_maps = raw_map_by_mode(
            patch,
            self.patch_lib,
            match_mode=self.match_mode,
            neighbor_radius=self.neighbor_radius,
        )  # [B, H, W]
        score_maps = select_score_map(raw_maps, self.score_stats, self.match_mode)
        scores = score_maps.amax(dim=(1, 2)).detach().cpu()
        return scores, score_maps

    def predict_tensor(self, sample):
        scores, score_maps = self.predict_batch_tensor(sample)
        return scores[0], score_maps[0]

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

    def predict_big_image(self, image_path, rows, cols, top_margin, bottom_margin, left_margin, right_margin, hori_gap, vert_gap):
        start = time.time()
        image = Image.open(image_path).convert("RGB")
        full_score = np.zeros((image.height, image.width), dtype=np.float32)
        tiles = split_big_image_by_geometry(
            image,
            rows=rows,
            cols=cols,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
            hori_gap=hori_gap,
            vert_gap=vert_gap,
        )
        tile_samples = [self.transform(tile.image).unsqueeze(0) for tile in tiles]
        batch = torch.cat(tile_samples, dim=0)  # [B, 3, image_h, image_w]
        scores, score_maps = self.predict_batch_tensor(batch)

        tile_rows = []
        for tile, score, score_map in zip(tiles, scores, score_maps):
            score_value = float(score.item())
            stitch_tile_score(full_score, score_map, tile.box)
            tile_rows.append({
                "tile_index": tile.index,
                "row": tile.row,
                "col": tile.col,
                "score": score_value,
                "x1": tile.box[0],
                "y1": tile.box[1],
                "x2": tile.box[2],
                "y2": tile.box[3],
            })
        elapsed_ms = (time.time() - start) * 1000
        score = float(full_score.max())
        return image, score, torch.from_numpy(full_score), elapsed_ms, tile_rows


def stitch_tile_score(full_score, tile_score_map, box):
    x1, y1, x2, y2 = box
    tile_score = tile_score_map.detach().cpu().float().numpy()
    tile_score = cv2.resize(tile_score, (x2 - x1, y2 - y1))
    full_score[y1:y2, x1:x2] = tile_score


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
    combined = cv2.hconcat([image_bgr, overlay])
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


def write_tile_scores_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "tile_index", "row", "col", "score", "x1", "y1", "x2", "y2"])
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
@click.option("--big-image", is_flag=True)
@click.option("--rows", default=5, type=int)
@click.option("--cols", default=3, type=int)
@click.option("--top-margin", default=626, type=int)
@click.option("--bottom-margin", default=626, type=int)
@click.option("--left-margin", default=3, type=int)
@click.option("--right-margin", default=3, type=int)
@click.option("--hori-gap", default=1, type=int)
@click.option("--vert-gap", default=1, type=int)
def cli_interface(model_path, image, input_path, output_dir, backbone, image_size, fmap_size, resize_method, out_indices, match_mode, neighbor_radius, big_image, rows, cols, top_margin, bottom_margin, left_margin, right_margin, hori_gap, vert_gap):
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
    score_rows = []
    all_tile_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        if big_image:
            src_image, score, score_map, elapsed_ms, tile_rows = predictor.predict_big_image(
                image_path,
                rows=rows,
                cols=cols,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                left_margin=left_margin,
                right_margin=right_margin,
                hori_gap=hori_gap,
                vert_gap=vert_gap,
            )
            for tile_row in tile_rows:
                tile_row["path"] = str(image_path)
                all_tile_rows.append(tile_row)
        else:
            src_image, score, score_map, elapsed_ms = predictor.predict_image(image_path)
        label = infer_label_from_path(image_path)
        result_path = save_heatmap_outputs(src_image, score_map, image_path, output_dir, label, score, elapsed_ms)
        score_rows.append({
            "path": str(image_path),
            "label": label,
            "score": score,
            "elapsed_ms": round(elapsed_ms, 2),
            "result_path": str(result_path),
        })
        print(f"{image_path}: label={label}, score={score:.4f}, elapsed_ms={elapsed_ms:.1f}")
    write_scores_csv(output_dir / "scores.csv", score_rows)
    if all_tile_rows:
        write_tile_scores_csv(output_dir / "tile_scores.csv", all_tile_rows)
    image_rocauc = write_metrics_json(output_dir / "metrics.json", score_rows)
    print(f"image_rocauc={image_rocauc:.4f}")


if __name__ == "__main__":
    cli_interface()
