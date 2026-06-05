# Standalone Predictor Match Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `global`, `same_row`, and `exact_position` matching modes plus `neighbor_radius > 0` support to the standalone PatchCore predictor.

**Architecture:** Keep the predictor standalone and focused. Add pure score-map helper functions that accept an extracted patch tensor `[1, C, H, W]` and a position-indexed memory bank `[H, W, N, C]`, returning a raw score map `[H, W]`. `PatchCorePredictor.predict_tensor()` delegates to these helpers, applies saved score stats only for `exact_position`, and the CLI exposes `--match-mode` and `--neighbor-radius`.

**Tech Stack:** Python, PyTorch, Click, pytest, existing `transfusion` conda environment.

---

## File Structure

- Modify: `indad/patchcore_predict_simple.py`
  - Add raw-map helper functions for the three matching modes.
  - Add `match_mode` argument to `PatchCorePredictor` and CLI.
  - Allow `neighbor_radius > 0` for `same_row` and `exact_position`.

- Modify: `tests/test_patchcore_predict_simple.py`
  - Add small tensor tests for `global`, `same_row`, and `exact_position` raw-map helpers.
  - Add test that score stats are applied only for `exact_position`.

---

### Task 1: Add raw-map helper tests

**Files:**
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Add failing imports and tests**

Append to `tests/test_patchcore_predict_simple.py`:

```python
from patchcore_predict_simple import (
    raw_map_exact_position,
    raw_map_global,
    raw_map_same_row,
    select_score_map,
)


def test_raw_map_global_matches_any_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])  # [1, C=1, H=1, W=2]
    patch_lib = torch.tensor([[[[5.0]], [[10.0]]]])  # [H=1, W=2, N=1, C=1]

    raw = raw_map_global(patch, patch_lib)

    assert torch.allclose(raw, torch.tensor([[5.0, 0.0]]))


def test_raw_map_exact_position_radius_zero_matches_same_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])
    patch_lib = torch.tensor([[[[5.0]], [[7.0]]]])

    raw = raw_map_exact_position(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[5.0, 3.0]]))


def test_raw_map_exact_position_radius_one_can_match_neighbor_position():
    patch = torch.tensor([[[[0.0, 10.0]]]])
    patch_lib = torch.tensor([[[[5.0]], [[10.0]]]])

    raw = raw_map_exact_position(patch, patch_lib, neighbor_radius=1)

    assert torch.allclose(raw, torch.tensor([[5.0, 0.0]]))


def test_raw_map_same_row_radius_zero_matches_same_row():
    patch = torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]])
    patch_lib = torch.tensor([
        [[[1.0]], [[11.0]]],
        [[[100.0]], [[31.0]]],
    ])

    raw = raw_map_same_row(patch, patch_lib, neighbor_radius=0)

    assert torch.allclose(raw, torch.tensor([[1.0, 1.0], [11.0, 1.0]]))


def test_select_score_map_applies_stats_only_for_exact_position():
    raw = torch.tensor([[3.0]])
    stats = {"baseline": torch.tensor([[1.0]]), "scale": torch.tensor([[2.0]])}

    exact = select_score_map(raw, stats, match_mode="exact_position")
    global_map = select_score_map(raw, stats, match_mode="global")

    assert torch.allclose(exact, torch.tensor([[1.0]]))
    assert torch.allclose(global_map, raw)
```

- [ ] **Step 2: Run tests and verify missing helper failure**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: FAIL with import errors for the new helper functions.

---

### Task 2: Implement raw-map helpers and predictor delegation

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add matching helper functions**

In `indad/patchcore_predict_simple.py`, add these functions after `normalize_vis_map(...)`:

```python

def _patch_to_hwc(patch):
    return patch.permute(0, 2, 3, 1).squeeze(0)


def raw_map_global(patch, patch_lib):
    patch_hwc = _patch_to_hwc(patch)
    H, W, C = patch_hwc.shape
    query = patch_hwc.reshape(-1, C)
    lib = patch_lib.reshape(-1, patch_lib.shape[-1])
    dist = torch.cdist(query, lib)
    return torch.min(dist, dim=1).values.reshape(H, W)


def raw_map_same_row(patch, patch_lib, neighbor_radius=0):
    patch_hwc = _patch_to_hwc(patch)
    H, W, N, C = patch_lib.shape
    row_lib = patch_lib.reshape(H, W * N, C)
    values = []
    r = int(neighbor_radius)
    for h in range(H):
        start = max(0, h - r)
        end = min(H, h + r + 1)
        candidates = row_lib[start:end].reshape(-1, C)
        dist = torch.cdist(patch_hwc[h], candidates)
        values.append(torch.min(dist, dim=1).values)
    return torch.stack(values, dim=0)


def raw_map_exact_position(patch, patch_lib, neighbor_radius=0):
    patch_hwc = _patch_to_hwc(patch)
    H, W, N, C = patch_lib.shape
    r = int(neighbor_radius)
    values = torch.empty(H, W, dtype=patch_hwc.dtype, device=patch_hwc.device)
    for h in range(H):
        h0 = max(0, h - r)
        h1 = min(H, h + r + 1)
        for w in range(W):
            w0 = max(0, w - r)
            w1 = min(W, w + r + 1)
            candidates = patch_lib[h0:h1, w0:w1].reshape(-1, C)
            dist = torch.cdist(patch_hwc[h, w].reshape(1, C), candidates)
            values[h, w] = torch.min(dist)
    return values


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
```

- [ ] **Step 2: Update `PatchCorePredictor.__init__`**

Add parameter:

```python
match_mode="exact_position"
```

Store:

```python
self.match_mode = match_mode
```

- [ ] **Step 3: Update `predict_tensor`**

Replace the current exact-position-only body with:

```python
    def predict_tensor(self, sample):
        patch = self.extract_patch(sample)
        raw_map = raw_map_by_mode(
            patch,
            self.patch_lib,
            match_mode=self.match_mode,
            neighbor_radius=self.neighbor_radius,
        )
        score_map = select_score_map(raw_map, self.score_stats, self.match_mode)
        score = torch.max(score_map).detach().cpu()
        return score, score_map
```

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: add standalone predictor match modes"
```

---

### Task 3: Wire CLI and smoke test

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add CLI option**

Add click option:

```python
@click.option("--match-mode", default="exact_position", type=click.Choice(["global", "same_row", "exact_position"]))
```

Update `cli_interface(...)` signature to include `match_mode`.

Pass into `PatchCorePredictor(...)`:

```python
match_mode=match_mode,
```

- [ ] **Step 2: Compile/help/test**

Run:

```bash
conda run -n transfusion python -m py_compile indad/patchcore_predict_simple.py
conda run -n transfusion python indad/patchcore_predict_simple.py --help
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: help shows `--match-mode`, tests PASS.

- [ ] **Step 3: Smoke test all three modes on one image**

Use the model and image known from prior smoke tests:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model outputs/patchcore_2604291557639_cv2_resnet18_23_32x96_3x256x768_fp32_76a28de6.ts2 \
  --image D:/python_project/dataset/LabelMe_171023_output/test/good/10_0_0.jpg \
  --output ./results-predict-simple-match-exact \
  --match-mode exact_position \
  --neighbor-radius 0

conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model outputs/patchcore_2604291557639_cv2_resnet18_23_32x96_3x256x768_fp32_76a28de6.ts2 \
  --image D:/python_project/dataset/LabelMe_171023_output/test/good/10_0_0.jpg \
  --output ./results-predict-simple-match-row \
  --match-mode same_row \
  --neighbor-radius 1

conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model outputs/patchcore_2604291557639_cv2_resnet18_23_32x96_3x256x768_fp32_76a28de6.ts2 \
  --image D:/python_project/dataset/LabelMe_171023_output/test/good/10_0_0.jpg \
  --output ./results-predict-simple-match-global \
  --match-mode global
```

Expected: each run writes `scores.csv`, `metrics.json`, and one combined heatmap image.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py
git commit -m "feat: expose predictor match mode CLI"
```

---

## Self-Review Checklist

- Spec coverage:
  - `global`, `same_row`, `exact_position`: Task 1/2 tests and helpers.
  - `neighbor_radius > 0`: Task 1 tests for exact_position radius and same_row helper supports radius.
  - score stats only exact_position: `select_score_map` test.
  - CLI exposes `--match-mode` and existing `--neighbor-radius`: Task 3.

- Placeholder scan:
  - No unresolved placeholders are intentionally left.

- Type consistency:
  - Raw map helpers return `[H, W]` tensors.
  - `match_mode` is one of `global`, `same_row`, `exact_position`.
  - `neighbor_radius` is an int used only by `same_row` and `exact_position`.
