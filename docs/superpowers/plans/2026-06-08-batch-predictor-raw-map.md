# Batch Predictor Raw Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standalone PatchCore predictor raw-map computation batch-aware so big-image tiles are processed as one `[B, C, H, W]` tensor instead of tile-by-tile.

**Architecture:** Keep public CLI behavior unchanged. Change raw-map helpers to accept `patch: [B, C, H, W]` and return `raw_maps: [B, H, W]`. Use `[H, W, B, C]` internally for exact-position alignment with `patch_lib: [H, W, N, C]`; use `[H, W*B, C]` for same-row; use `[B*H*W, C]` for global. Add `predict_batch_tensor()` and make both ordinary single-image inference and big-image tile inference call it.

**Tech Stack:** Python, PyTorch tensor ops, pytest, existing `transfusion` conda environment.

---

## File Structure

- Modify: `indad/patchcore_predict_simple.py`
  - Convert raw-map helpers to batch-aware versions.
  - Add `predict_batch_tensor()`.
  - Update `predict_tensor()` to wrap batch output.
  - Update `predict_big_image()` to stack all tiles into `[B, C, H, W]` and run one batch inference call.

- Modify: `tests/test_patchcore_predict_simple.py`
  - Update existing raw-map expected shapes from `[H, W]` to `[B, H, W]`.
  - Add batch tests with `B=2`.

---

### Task 1: Update raw-map tests for batch output

**Files:**
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Update existing helper tests to expect `[B, H, W]`**

Change existing raw-map assertions:

```python
assert torch.allclose(raw, torch.tensor([[5.0, 0.0]]))
```

to:

```python
assert torch.allclose(raw, torch.tensor([[[5.0, 0.0]]]))
```

Apply this pattern to:

- `test_raw_map_global_matches_any_position`
- `test_raw_map_exact_position_radius_zero_matches_same_position`
- `test_raw_map_exact_position_radius_one_can_match_neighbor_position`
- `test_raw_map_same_row_radius_zero_matches_same_row`
- `test_raw_map_same_row_radius_one_can_match_neighbor_row`

- [ ] **Step 2: Add batch-specific tests**

Append to `tests/test_patchcore_predict_simple.py`:

```python

def test_raw_map_exact_position_supports_batch_dimension():
    patch = torch.tensor([
        [[
            [0.0, 10.0],
        ]],
        [[
            [5.0, 12.0],
        ]],
    ])  # [B=2, C=1, H=1, W=2]
    patch_lib = torch.tensor([[[[0.0]], [[10.0]]]])

    raw = raw_map_exact_position(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[[0.0, 0.0]], [[5.0, 2.0]]]))


def test_raw_map_same_row_supports_batch_dimension():
    patch = torch.tensor([
        [[[0.0, 10.0]]],
        [[[2.0, 12.0]]],
    ])
    patch_lib = torch.tensor([[[[0.0]], [[10.0]]]])

    raw = raw_map_same_row(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]]]))


def test_raw_map_global_supports_batch_dimension():
    patch = torch.tensor([
        [[[0.0, 10.0]]],
        [[[2.0, 12.0]]],
    ])
    patch_lib = torch.tensor([[[[0.0]], [[10.0]]]])

    raw = raw_map_global(patch, patch_lib)

    assert torch.allclose(raw, torch.tensor([[[0.0, 0.0]], [[2.0, 2.0]]]))
```

- [ ] **Step 3: Run tests and verify failures**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: FAIL because raw-map helpers still return `[H, W]`.

---

### Task 2: Implement batch-aware raw-map helpers

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Replace `_patch_to_hwc` with batch-aware helper**

Replace:

```python
def _patch_to_hwc(patch):
    # patch: [1, C, H, W] -> [H, W, C]
    return patch.permute(0, 2, 3, 1).squeeze(0)
```

with:

```python
def _patch_to_hwbc(patch):
    # patch: [B, C, H, W] -> [H, W, B, C]
    return patch.permute(2, 3, 0, 1)
```

- [ ] **Step 2: Rewrite `raw_map_global`**

Replace its body with:

```python
    # patch: [B, C, H, W]
    # patch_lib: [H, W, N, C]
    B, C, H, W = patch.shape
    query = patch.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
    lib = patch_lib.reshape(-1, patch_lib.shape[-1])         # [H*W*N, C]
    dist = torch.cdist(query, lib)                            # [B*H*W, H*W*N]
    return torch.min(dist, dim=1).values.reshape(B, H, W)     # [B, H, W]
```

- [ ] **Step 3: Rewrite `raw_map_exact_position`**

Replace its body with:

```python
    patch_hwbc = _patch_to_hwbc(patch)  # [H, W, B, C]
    H, W, N, C = patch_lib.shape
    r = int(neighbor_radius)
    if r == 0:
        dist = torch.cdist(patch_hwbc, patch_lib)             # [H, W, B, N]
        raw_hwb = torch.min(dist, dim=-1).values              # [H, W, B]
        return raw_hwb.permute(2, 0, 1)                       # [B, H, W]

    K = 2 * r + 1
    lib_pad = torch.nn.functional.pad(
        patch_lib.permute(2, 3, 0, 1),
        (r, r, r, r),
        mode="replicate",
    ).permute(2, 3, 0, 1)
    lib_pad = lib_pad.unfold(0, K, 1).unfold(1, K, 1)
    lib_pad = lib_pad.permute(0, 1, 4, 5, 2, 3).contiguous()
    lib_pad = lib_pad.reshape(H, W, K * K * N, C)             # [H, W, K*K*N, C]
    dist = torch.cdist(patch_hwbc, lib_pad)                   # [H, W, B, K*K*N]
    raw_hwb = torch.min(dist, dim=-1).values                  # [H, W, B]
    return raw_hwb.permute(2, 0, 1)                           # [B, H, W]
```

- [ ] **Step 4: Rewrite `raw_map_same_row`**

Replace its body with:

```python
    patch_hwbc = _patch_to_hwbc(patch)                        # [H, W, B, C]
    H, W, N, C = patch_lib.shape
    query = patch_hwbc.reshape(H, W * patch_hwbc.shape[2], C) # [H, W*B, C]
    row_lib = patch_lib.reshape(H, W * N, C)                  # [H, W*N, C]
    r = int(neighbor_radius)
    if r == 0:
        dist = torch.cdist(query, row_lib)                    # [H, W*B, W*N]
        raw_h_wb = torch.min(dist, dim=2).values              # [H, W*B]
        return raw_h_wb.reshape(H, W, -1).permute(2, 0, 1)    # [B, H, W]

    K = 2 * r + 1
    lib_pad = torch.nn.functional.pad(
        row_lib.permute(1, 2, 0),
        (r, r),
        mode="replicate",
    ).permute(2, 0, 1)
    lib_pad = lib_pad.unfold(0, K, 1)
    lib_pad = lib_pad.permute(0, 3, 1, 2).contiguous()
    lib_pad = lib_pad.reshape(H, K * W * N, C)                # [H, K*W*N, C]
    dist = torch.cdist(query, lib_pad)                        # [H, W*B, K*W*N]
    raw_h_wb = torch.min(dist, dim=2).values                  # [H, W*B]
    return raw_h_wb.reshape(H, W, -1).permute(2, 0, 1)        # [B, H, W]
```

- [ ] **Step 5: Update `select_score_map` / `apply_score_stats` if needed**

Ensure `[B, H, W]` works with `[H, W]` stats by broadcasting:

```python
return torch.clamp_min((raw_map.float() - baseline.float()) / scale.float(), 0.0)
```

This already broadcasts. Do not change unless tests fail.

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: make predictor raw maps batch-aware"
```

---

### Task 3: Use batch inference in ordinary and big-image prediction

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add `predict_batch_tensor`**

Inside `PatchCorePredictor`, replace current `predict_tensor(...)` with:

```python
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
```

- [ ] **Step 2: Update `predict_big_image` to stack all tile samples**

Replace the tile loop in `predict_big_image` with:

```python
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
```

- [ ] **Step 3: Run full tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS.

---

### Task 4: Smoke test big-image batch inference

**Files:**
- No source changes expected unless smoke reveals a bug.

- [ ] **Step 1: Run big-image smoke**

Run:

```bash
rm -rf results-predict-big-smoke-batch
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model outputs/patchcore_2604291557639_cv2_resnet18_23_32x96_3x256x768_fp32_76a28de6.ts2 \
  --image D:/python_project/dataset/LabelMe_171023/test/bad/10.jpg \
  --output ./results-predict-big-smoke-batch \
  --big-image \
  --match-mode exact_position \
  --neighbor-radius 0
```

Expected: command succeeds and writes one big-image result.

- [ ] **Step 2: Verify row counts**

Run:

```bash
python - <<'PY'
from pathlib import Path
import csv
root=Path('results-predict-big-smoke-batch')
print((root/'scores.csv').exists())
print((root/'tile_scores.csv').exists())
print(len(list(csv.DictReader((root/'scores.csv').open(encoding='utf-8')))))
print(len(list(csv.DictReader((root/'tile_scores.csv').open(encoding='utf-8')))))
print(len(list((root/'heatmaps').glob('*.jpg'))))
PY
```

Expected:

```text
True
True
1
15
1
```

- [ ] **Step 3: Compare with previous sequential smoke if available**

If `results-predict-big-smoke/scores.csv` exists, compare score with new batch output. They should be very close.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "perf: batch big-image tile prediction"
```

---

## Self-Review Checklist

- Raw helpers accept `[B, C, H, W]` and return `[B, H, W]`.
- exact_position internally computes `[H, W, B]` before returning `[B, H, W]`.
- same_row uses query `[H, W*B, C]`.
- global uses query `[B*H*W, C]`.
- `predict_batch_tensor()` is the common path for single-image and big-image inference.
- Big-image mode builds one `[B, 3, H, W]` tile batch and predicts it in one call.
