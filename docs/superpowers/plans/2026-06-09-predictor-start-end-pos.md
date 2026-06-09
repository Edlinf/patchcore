# Predictor Start/End Position Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--start-pos` and `--end-pos` to standalone PatchCore predictor for ordinary image inference, matching training-side horizontal feature-map cropping behavior.

**Architecture:** Store `start_pos/end_pos` on `PatchCorePredictor`, crop the extracted patch tensor on feature-map width after feature concat, and crop the displayed source image for ordinary prediction outputs. Big-image mode explicitly rejects start/end in first version to avoid ambiguous semantics.

**Tech Stack:** Python, PyTorch, PIL/OpenCV, Click, pytest.

---

## File Structure

- Modify: `indad/patchcore_predict_simple.py`
  - Add `start_pos` and `end_pos` constructor/CLI args.
  - Crop patch tensor width dimension in `extract_patch()`.
  - Crop source image for ordinary prediction visualization.
  - Reject `--big-image` combined with non-zero start/end.

- Modify: `tests/test_patchcore_predict_simple.py`
  - Add tests for patch-width cropping and big-image validation behavior.

---

### Task 1: Add tests for start/end behavior

**Files:**
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Add tests**

Append:

```python

def test_predictor_crop_patch_width_by_start_end():
    predictor = PatchCorePredictor("dummy.ts", start_pos=16, end_pos=32)
    patch = torch.zeros(1, 4, 2, 8)
    cropped = predictor.crop_patch_width(patch)

    assert cropped.shape == torch.Size([1, 4, 2, 2])


def test_predictor_crop_source_image_for_visualization():
    predictor = PatchCorePredictor("dummy.ts", start_pos=2, end_pos=5)
    image = Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8), "RGB")

    cropped = predictor.crop_source_image(image)

    assert cropped.size == (3, 4)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py::test_predictor_crop_patch_width_by_start_end tests/test_patchcore_predict_simple.py::test_predictor_crop_source_image_for_visualization -v
```

Expected: FAIL because constructor args and helper methods do not exist.

---

### Task 2: Implement predictor cropping helpers

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Update constructor**

Add parameters:

```python
start_pos=0,
end_pos=0,
```

Store:

```python
self.start_pos = int(start_pos)
self.end_pos = int(end_pos)
```

- [ ] **Step 2: Add helper methods inside `PatchCorePredictor`**

Add before `extract_patch()`:

```python
    def crop_patch_width(self, patch):
        if self.start_pos != 0 or self.end_pos != 0:
            return patch[:, :, :, int(self.start_pos / 8):int(self.end_pos / 8)]
        return patch

    def crop_source_image(self, image):
        if self.start_pos != 0 or self.end_pos != 0:
            return image.crop((self.start_pos, 0, self.end_pos, image.height))
        return image
```

- [ ] **Step 3: Apply patch crop in `extract_patch()`**

Replace:

```python
return torch.cat(resized_maps, 1)
```

with:

```python
patch = torch.cat(resized_maps, 1)
return self.crop_patch_width(patch)
```

- [ ] **Step 4: Apply image crop in ordinary `predict_image()`**

Replace:

```python
return image, float(score.item()), score_map, elapsed_ms
```

with:

```python
return self.crop_source_image(image), float(score.item()), score_map, elapsed_ms
```

Do not crop big-image source images.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS.

---

### Task 3: Wire CLI and big-image validation

**Files:**
- Modify: `indad/patchcore_predict_simple.py`
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Add CLI options**

Add click options:

```python
@click.option("--start-pos", default=0, type=int)
@click.option("--end-pos", default=0, type=int)
```

Add parameters to `cli_interface(...)` and pass into `PatchCorePredictor(...)`.

- [ ] **Step 2: Reject big-image with start/end**

Inside `cli_interface`, after input validation, add:

```python
if big_image and (start_pos != 0 or end_pos != 0):
    raise click.UsageError("--start-pos/--end-pos are not supported with --big-image")
```

- [ ] **Step 3: Run compile/help/tests**

Run:

```bash
conda run -n transfusion python -m py_compile indad/patchcore_predict_simple.py
conda run -n transfusion python indad/patchcore_predict_simple.py --help
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: help shows `--start-pos` and `--end-pos`, tests PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py docs/superpowers/plans/2026-06-09-predictor-start-end-pos.md
git commit -m "feat: add predictor start/end position crop"
```

---

## Self-Review Checklist

- `start_pos/end_pos` defaults are 0 and preserve current behavior.
- Feature crop uses width dimension after feature concat: `[B, C, H, W] -> [B, C, H, W_crop]`.
- Visualization source image is cropped for ordinary image/dir prediction.
- Big-image mode rejects start/end for now.
- Tests cover helper behavior.
