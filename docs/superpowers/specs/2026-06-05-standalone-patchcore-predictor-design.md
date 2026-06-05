# Standalone PatchCore Predictor Design

Date: 2026-06-05

## Problem

Training code is used online and should not be coupled into a lightweight local inference script. The existing `indad/predict.py` evaluates a dataset through the training-side model classes. For local and future deployment-oriented validation, we need a simpler standalone PatchCore predictor class that can load trained PatchCore archives and run inference on one image or a flat image directory.

The design should reference the structure of `D:/python_project/011-patchcore_predict/patchcore_predict.py`, but use the current repository's PatchCore archive format and v4.0 score-normalization fields.

## Goals

Add a standalone prediction script:

```text
indad/patchcore_predict_simple.py
```

It should:

1. Define its own lightweight `PatchCorePredictor` class.
2. Avoid importing the training `PatchCore` class from `indad/models.py`.
3. Load PatchCore memory bank archives saved by current training code.
4. Support old archives with only parameter `0` and new archives with `patch_lib`, `score_baseline`, `score_scale`, and `recommended_pixel_threshold`.
5. Use local backbone weights from `hub/checkpoints` via `torch.hub.set_dir(project_root / "hub")`.
6. Support single-image prediction and flat-directory batch prediction.
7. Default to `neighbor_radius=0` for simple exact-position matching and lower memory use.
8. Write `scores.csv` and heatmap/overlay images.

## Non-Goals

First version does not implement:

- training;
- dataset evaluation / ROC calculation;
- ONNX or FAISS backend;
- large composite-image splitting;
- heatmap stitching;
- `same_row` or `global` matching;
- service/API wrapper.

## Model Loading

The predictor loads a TorchScript archive with:

```python
ts = torch.jit.load(model_path, map_location="cpu")
params = dict(ts.named_parameters())
```

Archive compatibility:

```text
new archive:
    patch_lib
    score_baseline
    score_scale
    recommended_pixel_threshold

old archive:
    0
```

Loading rule:

```python
patch_lib = params["patch_lib"] if "patch_lib" in params else params["0"]
```

If score stats exist, they are stored and applied during prediction when shapes match. If stats are missing or mismatched, the predictor falls back to raw scores.

## Backbone Loading

The predictor uses `timm.create_model(...)` like training code, but in the standalone script.

Before creating the model:

```python
project_root = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(project_root / "hub"))
```

This uses local checkpoints under:

```text
hub/checkpoints
```

Known local files include:

```text
resnet18-5c106cde.pth
resnet18-5c106cde.pt
resnet50_a1_0-14fe96d1.pth
wide_resnet50_racm-8234f177.pth
mobilenetv2_100_ra-b33bc2c4.pth
```

Default backbone:

```text
resnet18
```

Default feature indices:

```text
2,3
```

## Model Metadata

The script should try to parse current model filenames, using logic compatible with `indad/predict.py`:

```text
patchcore_<job>_<resize_method>_<backbone>_23_<fmap_size>_<image_shape>_fp32_<hash>.ts
```

If parsing succeeds, infer:

```text
resize_method
backbone
out_indices
fmap_size
image_size
```

If parsing fails, allow manual flags:

```bash
--backbone resnet18
--image-size 1280,128
--fmap-size 160,16
--resize-method cv2
--out-indices 2,3
```

## Image Preprocessing

For one input image:

```text
PIL RGB image
 -> resize by cv2 or torchvision-compatible adaptive resize
 -> ToTensor
 -> ImageNet normalize
 -> tensor [1, 3, H, W]
```

Use local constants equivalent to training:

```python
IMAGENET_MEAN = torch.tensor([.485, .456, .406])
IMAGENET_STD = torch.tensor([.229, .224, .225])
```

The script may reuse `Cv2AdaptiveResize` from `indad/data.py` if doing so does not pull in training model dependencies. It should not import `PatchCore` from `models.py`.

## Feature Extraction

`PatchCorePredictor` constructs a frozen feature extractor:

```python
self.feature_extractor = timm.create_model(
    backbone_name,
    out_indices=out_indices,
    features_only=True,
    pretrained=True,
)
self.feature_extractor.eval()
```

Feature maps are passed through:

```python
AvgPool2d(3, stride=1)
AdaptiveAvgPool2d(largest_fmap_size)
concat along channel dimension
```

This mirrors training `PatchCore.fit/predict` logic.

## Exact-Position Prediction

For each image:

```text
patch: [1, C, H, W]
patch_lib: [H, W, N, C]
```

With `neighbor_radius=0`:

```python
patch = patch.permute(0, 2, 3, 1).squeeze(0).unsqueeze(2)  # [H, W, 1, C]
dist = torch.cdist(patch, patch_lib)                       # [H, W, 1, N]
raw_map = min(dist, dim=-1).values.reshape(H, W)
```

Image score:

```python
score = raw_or_norm_map.max()
```

If score stats are available:

```python
norm_map = (raw_map - score_baseline) / score_scale
norm_map = clamp_min(norm_map, 0)
```

The normalized map becomes the output score map. Otherwise raw map is used.

## CLI

Single image:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model D:/path/to/model.ts \
  --image D:/path/to/image.jpg \
  --output ./results-predict-simple
```

Directory batch:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model D:/path/to/model.ts \
  --input D:/path/to/images \
  --output ./results-predict-simple
```

Optional flags:

```text
--backbone resnet18
--image-size 1280,128
--fmap-size 160,16
--resize-method cv2
--out-indices 2,3
--neighbor-radius 0
```

First version supports only a flat directory scan for:

```text
*.jpg, *.jpeg, *.png, *.bmp
```

No recursion.

## Outputs

For each run:

```text
output/
    scores.csv
    heatmaps/
        <stem>_heatmap.jpg
        <stem>_overlay.jpg
```

`scores.csv` columns:

```text
path,score,elapsed_ms,heatmap_path,overlay_path
```

Heatmap behavior:

- Normalize the score map to 0-255 for visualization.
- Apply OpenCV JET colormap.
- Save heatmap alone and overlay with resized input image.

## Acceptance Criteria

- Script compiles with `py_compile`.
- `--help` shows single-image, directory, model metadata, and neighbor-radius options.
- Can load a current PatchCore archive without importing `models.PatchCore`.
- Can run on one image and produce `scores.csv` plus heatmap/overlay.
- Can run on a flat directory and produce one CSV row per image.
- Default `neighbor_radius` is 0.
- Missing score stats fallback to raw score instead of failing.
