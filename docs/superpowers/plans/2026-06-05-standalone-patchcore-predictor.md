# Standalone PatchCore Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simple standalone PatchCore predictor script with its own `PatchCorePredictor` class, supporting single-image and flat-directory inference without importing `models.PatchCore`.

**Architecture:** Implement one self-contained CLI script under `indad/patchcore_predict_simple.py`. It loads PatchCore TorchScript archives, constructs a frozen `timm` feature extractor using local `hub/checkpoints`, preprocesses images, runs exact-position nearest-neighbor inference with default `neighbor_radius=0`, optionally applies saved score stats, and writes CSV plus heatmap/overlay outputs. Tests cover parsing/archive/preprocessing helpers without requiring a real backbone load.

**Tech Stack:** Python, Click, PyTorch, timm, torchvision transforms, PIL, OpenCV, NumPy, pytest, existing `transfusion` conda environment.

---

## File Structure

- Create: `indad/patchcore_predict_simple.py`
  - Standalone `PatchCorePredictor` class and CLI.
  - Does not import `PatchCore` from `indad/models.py`.
  - May import lightweight resize constants/classes from `indad/data.py` if useful.

- Create: `tests/test_patchcore_predict_simple.py`
  - Unit tests for model filename parsing fallback, archive loading, image collection, and score normalization helper behavior.

---

### Task 1: Add standalone predictor scaffold and archive loading tests

**Files:**
- Create: `indad/patchcore_predict_simple.py`
- Create: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_patchcore_predict_simple.py`:

```python
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INDAD = ROOT / "indad"
if str(INDAD) not in sys.path:
    sys.path.insert(0, str(INDAD))

from patchcore_predict_simple import collect_images, load_patchcore_archive_simple, parse_pair
from models import save_patchcore_archive, save_tensor


def test_parse_pair_reads_width_height():
    assert parse_pair("1280,128") == [1280, 128]
    assert parse_pair("48x16") == [48, 16]


def test_collect_images_reads_flat_directory_only(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.jpg").write_bytes(b"x")

    images = collect_images(tmp_path)

    assert [p.name for p in images] == ["a.jpg", "b.png"]


def test_load_patchcore_archive_simple_supports_old_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    save_tensor(str(tmp_path), "old.ts", patch_lib)

    loaded, stats = load_patchcore_archive_simple(tmp_path / "old.ts")

    assert torch.allclose(loaded, patch_lib)
    assert stats is None


def test_load_patchcore_archive_simple_supports_new_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    stats = {
        "baseline": torch.ones(2, 3),
        "scale": torch.full((2, 3), 2.0),
        "recommended_pixel_threshold": torch.tensor(3.5),
    }
    save_patchcore_archive(str(tmp_path), "new.ts", patch_lib, stats)

    loaded, loaded_stats = load_patchcore_archive_simple(tmp_path / "new.ts")

    assert torch.allclose(loaded, patch_lib)
    assert torch.allclose(loaded_stats["baseline"], stats["baseline"])
    assert torch.allclose(loaded_stats["scale"], stats["scale"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing function imports.

- [ ] **Step 3: Create initial `indad/patchcore_predict_simple.py`**

Create the file with:

```python
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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS for 4 tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: add standalone PatchCore predictor scaffold"
```

---

### Task 2: Implement metadata parsing, preprocessing, and predictor class

**Files:**
- Modify: `indad/patchcore_predict_simple.py`
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Add tests for score normalization and metadata parsing**

Append to `tests/test_patchcore_predict_simple.py`:

```python
from patchcore_predict_simple import apply_score_stats, parse_model_info_simple


def test_apply_score_stats_clamps_negative_values():
    raw = torch.tensor([[1.0, 5.0]])
    stats = {
        "baseline": torch.tensor([[2.0, 1.0]]),
        "scale": torch.tensor([[1.0, 2.0]]),
    }

    out = apply_score_stats(raw, stats)

    assert torch.allclose(out, torch.tensor([[0.0, 2.0]]))


def test_parse_model_info_simple_reads_current_filename():
    info = parse_model_info_simple("patchcore_job_cv2_resnet18_23_16x48_3x128x384_fp32_abcd1234.ts")

    assert info["method"] == "patchcore"
    assert info["resize_method"] == "cv2"
    assert info["backbone"] == "resnet18"
    assert info["out_indices"] == [2, 3]
    assert info["fmap_size"] == [16, 48]
    assert info["image_size"] == [128, 384]
```

- [ ] **Step 2: Implement helpers and predictor class**

Add to `indad/patchcore_predict_simple.py` before the `if __name__` block:

```python

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
```

- [ ] **Step 3: Run tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: add standalone PatchCore predictor core"
```

---

### Task 3: Implement image prediction, outputs, and CLI

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add prediction/output methods**

Add these methods inside `PatchCorePredictor`:

```python
    def predict_tensor(self, sample):
        patch = self.extract_patch(sample)
        patch = patch.permute(0, 2, 3, 1).squeeze(0).unsqueeze(2)
        H, W, N, C = self.patch_lib.shape
        if self.neighbor_radius != 0:
            raise ValueError("simple predictor only supports neighbor_radius=0 in first version")
        dist = torch.cdist(patch, self.patch_lib)
        raw_map = torch.min(dist, dim=-1).values.reshape(H, W)
        score_map = apply_score_stats(raw_map, self.score_stats)
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
```

Add output helpers at module level:

```python

def save_heatmap_outputs(image, score_map, image_path, output_dir):
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    heat = normalize_vis_map(score_map)
    heat = cv2.resize(heat, image.size)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat_color, 0.5, image_bgr, 0.5, 0)
    heatmap_path = heatmap_dir / f"{stem}_heatmap.jpg"
    overlay_path = heatmap_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(heatmap_path), heat_color)
    cv2.imwrite(str(overlay_path), overlay)
    return heatmap_path, overlay_path


def write_scores_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "score", "elapsed_ms", "heatmap_path", "overlay_path"])
        writer.writeheader()
        writer.writerows(rows)
```

Add CLI at bottom replacing the `pass` block:

```python
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
@click.option("--neighbor-radius", default=0, type=int)
def cli_interface(model_path, image, input_path, output_dir, backbone, image_size, fmap_size, resize_method, out_indices, neighbor_radius):
    if image is None and input_path is None:
        raise click.UsageError("Provide --image or --input")
    info = None
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
        neighbor_radius=neighbor_radius,
        output_dir=output_dir,
    ).load()

    images = [image] if image is not None else collect_images(input_path)
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        src_image, score, score_map, elapsed_ms = predictor.predict_image(image_path)
        heatmap_path, overlay_path = save_heatmap_outputs(src_image, score_map, image_path, output_dir)
        rows.append({
            "path": str(image_path),
            "score": score,
            "elapsed_ms": round(elapsed_ms, 2),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
        })
        print(f"{image_path}: score={score:.4f}, elapsed_ms={elapsed_ms:.1f}")
    write_scores_csv(output_dir / "scores.csv", rows)


if __name__ == "__main__":
    cli_interface()
```

- [ ] **Step 2: Compile and help-check**

Run:

```bash
conda run -n transfusion python -m py_compile indad/patchcore_predict_simple.py
conda run -n transfusion python indad/patchcore_predict_simple.py --help
```

Expected: compile passes and help lists `--image`, `--input`, `--neighbor-radius`.

- [ ] **Step 3: Run unit tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: add standalone PatchCore image prediction CLI"
```

---

### Task 4: Manual smoke test with available model/image

**Files:**
- No source changes expected unless smoke test reveals a bug.

- [ ] **Step 1: Locate local model and image**

Run:

```bash
python - <<'PY'
from pathlib import Path
print('models')
for p in Path('outputs-grid-test-fast').rglob('*.ts2'):
    print(p)
for p in Path('outputs-grid-test').rglob('*.ts2'):
    print(p)
for p in Path('results').rglob('*.ts'):
    print(p)
print('images')
for p in Path('D:/python_project/dataset/LabelMe_171023_grid_3x5_gap1/test/good').glob('*.jpg'):
    print(p)
    break
PY
```

- [ ] **Step 2: Run single-image smoke if model exists**

Run with a model path from Step 1:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model <model_path> \
  --image D:/python_project/dataset/LabelMe_171023_grid_3x5_gap1/test/good/1_00.jpg \
  --output ./results-predict-simple-smoke
```

Expected: `scores.csv`, heatmap, overlay.

If no parseable model exists, skip and report missing prerequisite.

- [ ] **Step 3: Run directory smoke if single-image works**

Run:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model <model_path> \
  --input D:/python_project/dataset/LabelMe_171023_grid_3x5_gap1/test/good \
  --output ./results-predict-simple-smoke-dir
```

Expected: one CSV row per image in that flat directory.

- [ ] **Step 4: Commit fixes only if needed**

If smoke test required source changes:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "fix: stabilize standalone PatchCore predictor smoke test"
```

---

## Self-Review Checklist

- Spec coverage:
  - Standalone predictor class: Task 2/3.
  - Does not import `models.PatchCore`: Task 1/2/3 code content.
  - Local hub/checkpoints: Task 2 `load()`.
  - Old/new archive loading: Task 1.
  - Single image and directory batch: Task 3.
  - Default neighbor radius 0: Task 3 CLI and class default.
  - scores.csv/heatmap/overlay: Task 3.

- Placeholder scan:
  - No unresolved placeholders are intentionally left in this plan.

- Type consistency:
  - `image_size` and `fmap_size` are `[width, height]` pairs for CLI input, matching existing filename parse behavior.
  - `score_stats` is either `None` or dict with `baseline`, `scale`, `recommended_pixel_threshold`.
