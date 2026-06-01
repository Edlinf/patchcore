# Position-Normalized PatchCore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leave-one-out median/MAD position-normalized scoring to `PatchCore` `exact_position` inference while preserving old model compatibility.

**Architecture:** Put score-normalization math in a small pure helper module so it can be tested without loading a backbone. `PatchCore.fit()` computes per-position stats after coreset sampling and saves them alongside/inside the existing TorchScript archive. `PatchCore.predict()` computes the existing raw distance map, then uses normalized scores only when compatible stats are loaded and `match_mode == "exact_position"`.

**Tech Stack:** Python 3, PyTorch 1.9, torchvision/PIL, pytest-style tests, existing TorchScript archive saving/loading.

---

## File Structure

- Create: `indad/patchcore_normalization.py`
  - Pure tensor functions for leave-one-out distances, robust median/MAD stats, scale guards, and applying normalization.
  - No dependency on `timm`, datasets, file I/O, or model classes.

- Modify: `indad/models.py`
  - Import the new helper functions.
  - Extend model archive saving to register `patch_lib`, `score_baseline`, `score_scale`, and `recommended_pixel_threshold` when stats are available.
  - Add `PatchCore` constructor options for score normalization.
  - Compute stats in `PatchCore.fit()` after coreset sampling.
  - Load embedded stats from new archives, or fall back to raw scoring for old archives.
  - Normalize raw distance maps in `PatchCore.predict()` only for `exact_position`.

- Modify: `indad/run.py`
  - Preserve compatibility with new archive format. No CLI option is required for the first implementation.

- Modify: `indad/run-yml.py`
  - Preserve compatibility with new archive format. No config-template change is required for the first implementation.

- Create: `tests/test_patchcore_normalization.py`
  - Unit tests for leave-one-out, robust stats, scale floor, normalization, and disabled behavior with too few OK patches.

- Create: `tests/test_patchcore_archive_loading.py`
  - Unit tests for saving/loading new archives and loading old single-parameter archives.

---

### Task 1: Add pure normalization helpers

**Files:**
- Create: `indad/patchcore_normalization.py`
- Test: `tests/test_patchcore_normalization.py`

- [ ] **Step 1: Create the failing tests**

Create `tests/test_patchcore_normalization.py` with this content:

```python
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDAD = os.path.join(ROOT, "indad")
if INDAD not in sys.path:
    sys.path.insert(0, INDAD)

from patchcore_normalization import (
    apply_position_normalization,
    compute_position_score_stats,
    leave_one_out_nearest_distances,
)


def test_leave_one_out_nearest_distances_excludes_self_match():
    pos_lib = torch.tensor([
        [0.0],
        [2.0],
        [5.0],
    ])

    distances = leave_one_out_nearest_distances(pos_lib)

    assert torch.allclose(distances, torch.tensor([2.0, 2.0, 3.0]))


def test_compute_position_score_stats_uses_median_mad_and_scale_floor():
    patch_lib = torch.tensor([
        [
            [[0.0], [2.0], [5.0], [9.0]],
            [[0.0], [10.0], [20.0], [30.0]],
        ]
    ])  # [H=1, W=2, N=4, C=1]

    stats = compute_position_score_stats(
        patch_lib,
        min_train_patches=4,
        scale_floor_quantile=0.5,
        absolute_eps=0.01,
        smooth_scale=False,
        threshold_quantile=0.75,
    )

    assert set(stats.keys()) == {
        "baseline",
        "scale",
        "recommended_pixel_threshold",
        "loo_distances",
        "loo_z",
    }
    assert stats["baseline"].shape == torch.Size([1, 2])
    assert stats["scale"].shape == torch.Size([1, 2])
    assert torch.all(stats["scale"] >= 0.01)
    assert stats["recommended_pixel_threshold"].ndim == 0


def test_apply_position_normalization_clamps_negative_scores_to_zero():
    raw_map = torch.tensor([
        [1.0, 4.0],
        [7.0, 8.0],
    ])
    baseline = torch.tensor([
        [2.0, 2.0],
        [5.0, 6.0],
    ])
    scale = torch.tensor([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    norm_map = apply_position_normalization(raw_map, baseline, scale, clamp_min_zero=True)

    assert torch.allclose(norm_map, torch.tensor([
        [0.0, 1.0],
        [1.0, 2.0],
    ]))


def test_compute_position_score_stats_rejects_too_few_training_patches():
    patch_lib = torch.zeros(2, 2, 3, 4)

    with pytest.raises(ValueError, match="at least 4"):
        compute_position_score_stats(patch_lib, min_train_patches=4)
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'patchcore_normalization'`.

- [ ] **Step 3: Implement the helper module**

Create `indad/patchcore_normalization.py` with this content:

```python
import torch
import torch.nn.functional as F


def leave_one_out_nearest_distances(pos_lib: torch.Tensor) -> torch.Tensor:
    """Return each patch's nearest-neighbor distance excluding itself.

    Args:
        pos_lib: Tensor shaped [N, C] for one spatial position.

    Returns:
        Tensor shaped [N].
    """
    if pos_lib.ndim != 2:
        raise ValueError(f"pos_lib must be [N, C], got shape {tuple(pos_lib.shape)}")
    if pos_lib.shape[0] < 2:
        raise ValueError("leave-one-out distances need at least 2 patches")

    values = pos_lib.float()
    distances = torch.cdist(values, values)
    diag = torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    distances = distances.masked_fill(diag, float("inf"))
    return torch.min(distances, dim=1).values


def _median_mad(values: torch.Tensor):
    baseline = torch.median(values)
    mad = torch.median(torch.abs(values - baseline))
    scale = 1.4826 * mad
    return baseline, scale


def _smooth_2d_map(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return values
    if kernel_size % 2 == 0:
        raise ValueError("smooth_kernel must be odd")

    x = values.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    weight = torch.ones(1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device)
    weight = weight / weight.numel()
    return F.conv2d(x, weight).squeeze(0).squeeze(0)


def compute_position_score_stats(
    patch_lib: torch.Tensor,
    min_train_patches: int = 4,
    scale_floor_quantile: float = 0.2,
    scale_cap_quantile=None,
    absolute_eps: float = 1e-6,
    smooth_scale: bool = True,
    smooth_kernel: int = 3,
    threshold_quantile: float = 0.999,
):
    """Compute robust per-position PatchCore score-normalization stats.

    Args:
        patch_lib: Tensor shaped [H, W, N, C].

    Returns:
        Dict with baseline [H, W], scale [H, W], scalar threshold,
        leave-one-out distances [H, W, N], and normalized LOO scores [H, W, N].
    """
    if patch_lib.ndim != 4:
        raise ValueError(f"patch_lib must be [H, W, N, C], got shape {tuple(patch_lib.shape)}")

    H, W, N, _ = patch_lib.shape
    if N < min_train_patches:
        raise ValueError(
            f"position normalization needs at least {min_train_patches} training patches, got {N}"
        )

    baseline = torch.empty(H, W, dtype=torch.float32)
    scale = torch.empty(H, W, dtype=torch.float32)
    loo_distances = torch.empty(H, W, N, dtype=torch.float32)

    for h in range(H):
        for w in range(W):
            distances = leave_one_out_nearest_distances(patch_lib[h, w].float()).cpu()
            loo_distances[h, w] = distances
            baseline[h, w], scale[h, w] = _median_mad(distances)

    valid_scale = scale[torch.isfinite(scale)]
    if valid_scale.numel() == 0:
        scale_floor = torch.tensor(float(absolute_eps), dtype=torch.float32)
    else:
        scale_floor = torch.quantile(valid_scale, scale_floor_quantile)
        scale_floor = torch.maximum(scale_floor, torch.tensor(float(absolute_eps), dtype=torch.float32))

    scale = torch.maximum(scale, scale_floor)

    if scale_cap_quantile is not None:
        scale_cap = torch.quantile(scale, scale_cap_quantile)
        scale = torch.minimum(scale, scale_cap)
        scale = torch.maximum(scale, scale_floor)

    if smooth_scale:
        scale = _smooth_2d_map(scale, smooth_kernel)
        scale = torch.maximum(scale, scale_floor)

    loo_z = (loo_distances - baseline.unsqueeze(-1)) / scale.unsqueeze(-1)
    loo_z = torch.clamp_min(loo_z, 0.0)
    recommended_pixel_threshold = torch.quantile(loo_z.reshape(-1), threshold_quantile)

    return {
        "baseline": baseline,
        "scale": scale,
        "recommended_pixel_threshold": recommended_pixel_threshold,
        "loo_distances": loo_distances,
        "loo_z": loo_z,
    }


def apply_position_normalization(
    raw_map: torch.Tensor,
    baseline: torch.Tensor,
    scale: torch.Tensor,
    clamp_min_zero: bool = True,
) -> torch.Tensor:
    if raw_map.shape != baseline.shape or raw_map.shape != scale.shape:
        raise ValueError(
            "raw_map, baseline, and scale must have matching shapes: "
            f"raw={tuple(raw_map.shape)}, baseline={tuple(baseline.shape)}, scale={tuple(scale.shape)}"
        )

    norm_map = (raw_map.float() - baseline.float()) / scale.float()
    if clamp_min_zero:
        norm_map = torch.clamp_min(norm_map, 0.0)
    return norm_map
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py -v
```

Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add indad/patchcore_normalization.py tests/test_patchcore_normalization.py
git commit -m "feat: add PatchCore score normalization helpers"
```

---

### Task 2: Add PatchCore archive save/load helpers

**Files:**
- Modify: `indad/models.py:39-49`
- Test: `tests/test_patchcore_archive_loading.py`

- [ ] **Step 1: Create archive loading tests**

Create `tests/test_patchcore_archive_loading.py` with this content:

```python
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDAD = os.path.join(ROOT, "indad")
if INDAD not in sys.path:
    sys.path.insert(0, INDAD)

from models import load_patchcore_archive, save_patchcore_archive, save_tensor


def test_load_patchcore_archive_reads_old_single_parameter_archive(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    save_tensor(str(tmp_path), "old.ts", patch_lib)

    loaded_patch_lib, stats = load_patchcore_archive(str(tmp_path / "old.ts"))

    assert torch.allclose(loaded_patch_lib, patch_lib)
    assert stats is None


def test_load_patchcore_archive_reads_new_archive_with_embedded_stats(tmp_path):
    patch_lib = torch.randn(2, 3, 4, 5)
    stats = {
        "baseline": torch.ones(2, 3),
        "scale": torch.full((2, 3), 2.0),
        "recommended_pixel_threshold": torch.tensor(3.5),
    }
    save_patchcore_archive(str(tmp_path), "new.ts", patch_lib, stats)

    loaded_patch_lib, loaded_stats = load_patchcore_archive(str(tmp_path / "new.ts"))

    assert torch.allclose(loaded_patch_lib, patch_lib)
    assert torch.allclose(loaded_stats["baseline"], stats["baseline"])
    assert torch.allclose(loaded_stats["scale"], stats["scale"])
    assert torch.allclose(
        loaded_stats["recommended_pixel_threshold"],
        stats["recommended_pixel_threshold"],
    )
```

- [ ] **Step 2: Run the archive tests and verify they fail because helpers are missing**

Run:

```bash
python -m pytest tests/test_patchcore_archive_loading.py -v
```

Expected: FAIL with `ImportError: cannot import name 'load_patchcore_archive'`.

- [ ] **Step 3: Modify imports in `indad/models.py`**

In `indad/models.py`, replace the existing import block:

```python
from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
```

with:

```python
from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
from patchcore_normalization import (
    apply_position_normalization,
    compute_position_score_stats,
)
```

- [ ] **Step 4: Replace the current archive helpers**

In `indad/models.py`, replace lines 39-49:

```python
class Module(nn.Module):
    pass
    
def save_tensor(results_dir,filename,x):
    path = os.path.join(results_dir,filename)
    m = Module()
    par = nn.Parameter(x)
    m.register_parameter("0",par)
    tensors = torch.jit.script(m)
    tensors.save(path)
```

with:

```python
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
```

- [ ] **Step 5: Run the archive tests and verify they pass**

Run:

```bash
python -m pytest tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for both tests.

- [ ] **Step 6: Run normalization tests again**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all 6 tests.

- [ ] **Step 7: Commit**

Run:

```bash
git add indad/models.py tests/test_patchcore_archive_loading.py
git commit -m "feat: support PatchCore archives with score stats"
```

---

### Task 3: Compute and save normalization stats during PatchCore training

**Files:**
- Modify: `indad/models.py:393-500`
- Test: `tests/test_patchcore_archive_loading.py`

- [ ] **Step 1: Add a training-save test for stats shape**

Append this test to `tests/test_patchcore_archive_loading.py`:

```python

def test_new_archive_preserves_position_stat_shapes(tmp_path):
    patch_lib = torch.randn(4, 5, 6, 7)
    stats = {
        "baseline": torch.randn(4, 5),
        "scale": torch.ones(4, 5),
        "recommended_pixel_threshold": torch.tensor(2.25),
    }

    save_patchcore_archive(str(tmp_path), "patch_lib.ts", patch_lib, stats)
    loaded_patch_lib, loaded_stats = load_patchcore_archive(str(tmp_path / "patch_lib.ts"))

    assert loaded_patch_lib.shape == torch.Size([4, 5, 6, 7])
    assert loaded_stats["baseline"].shape == torch.Size([4, 5])
    assert loaded_stats["scale"].shape == torch.Size([4, 5])
    assert loaded_stats["recommended_pixel_threshold"].shape == torch.Size([])
```

- [ ] **Step 2: Run the new test and verify it passes before model integration**

Run:

```bash
python -m pytest tests/test_patchcore_archive_loading.py::test_new_archive_preserves_position_stat_shapes -v
```

Expected: PASS. This confirms the archive format is ready before wiring it into `PatchCore.fit()`.

- [ ] **Step 3: Extend `PatchCore.__init__` with normalization options**

In `indad/models.py`, inside `PatchCore.__init__`, after the existing `match_mode` argument, add these parameters:

```python
        score_normalization_enabled: bool = True,
        score_normalization_min_train_patches: int = 4,
        score_normalization_scale_floor_quantile: float = 0.2,
        score_normalization_scale_cap_quantile = None,
        score_normalization_smooth_scale: bool = True,
        score_normalization_smooth_kernel: int = 3,
        score_normalization_threshold_quantile: float = 0.999,
        score_normalization_clamp_min_zero: bool = True,
```

The full end of the signature should look like:

```python
        jobini = None,
        match_mode: str = "exact_position", # 推理匹配方式: global | same_row | exact_position，训练库统一按 exact_position 保存
        score_normalization_enabled: bool = True,
        score_normalization_min_train_patches: int = 4,
        score_normalization_scale_floor_quantile: float = 0.2,
        score_normalization_scale_cap_quantile = None,
        score_normalization_smooth_scale: bool = True,
        score_normalization_smooth_kernel: int = 3,
        score_normalization_threshold_quantile: float = 0.999,
        score_normalization_clamp_min_zero: bool = True,
    ):
```

- [ ] **Step 4: Store normalization options on the instance**

In `PatchCore.__init__`, after:

```python
        self.match_mode = match_mode
```

add:

```python
        self.score_normalization_enabled = score_normalization_enabled
        self.score_normalization_min_train_patches = score_normalization_min_train_patches
        self.score_normalization_scale_floor_quantile = score_normalization_scale_floor_quantile
        self.score_normalization_scale_cap_quantile = score_normalization_scale_cap_quantile
        self.score_normalization_smooth_scale = score_normalization_smooth_scale
        self.score_normalization_smooth_kernel = score_normalization_smooth_kernel
        self.score_normalization_threshold_quantile = score_normalization_threshold_quantile
        self.score_normalization_clamp_min_zero = score_normalization_clamp_min_zero
        self.score_stats = None
```

- [ ] **Step 5: Compute stats after coreset sampling in `PatchCore.fit()`**

In `PatchCore.fit()`, replace the final line:

```python
        save_tensor(self.results_dir,'patch_lib.ts',self.patch_lib)
```

with:

```python
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
```

- [ ] **Step 6: Update `PatchCore.get_parameters()`**

In `PatchCore.get_parameters()`, add these fields to the dictionary:

```python
            "score_normalization_enabled": self.score_normalization_enabled,
            "score_normalization_min_train_patches": self.score_normalization_min_train_patches,
            "score_normalization_scale_floor_quantile": self.score_normalization_scale_floor_quantile,
            "score_normalization_scale_cap_quantile": self.score_normalization_scale_cap_quantile,
            "score_normalization_smooth_scale": self.score_normalization_smooth_scale,
            "score_normalization_smooth_kernel": self.score_normalization_smooth_kernel,
            "score_normalization_threshold_quantile": self.score_normalization_threshold_quantile,
            "score_normalization_clamp_min_zero": self.score_normalization_clamp_min_zero,
            "score_normalization_has_stats": self.score_stats is not None,
```

The method should still include the existing fields `f_coreset`, `n_reweight`, `start_pos`, `end_pos`, and `match_mode`.

- [ ] **Step 7: Run tests**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all tests.

- [ ] **Step 8: Commit**

Run:

```bash
git add indad/models.py tests/test_patchcore_archive_loading.py
git commit -m "feat: compute PatchCore position score stats"
```

---

### Task 4: Load stats and normalize exact-position inference

**Files:**
- Modify: `indad/models.py:502-624`
- Test: `tests/test_patchcore_archive_loading.py`

- [ ] **Step 1: Add a test for `PatchCore.load()` old-model fallback**

Append this test to `tests/test_patchcore_archive_loading.py`:

```python

def test_patchcore_load_old_archive_leaves_score_stats_none(tmp_path):
    from models import PatchCore

    patch_lib = torch.randn(2, 2, 4, 3)
    save_tensor(str(tmp_path), "old.ts", patch_lib)

    model = PatchCore.__new__(PatchCore)
    model.score_stats = "not cleared"
    result = PatchCore.load(model, str(tmp_path / "old.ts"), [2, 2])

    assert result is True
    assert torch.allclose(model.patch_lib, patch_lib)
    assert model.score_stats is None
```

- [ ] **Step 2: Add a test for `PatchCore.load()` new stats**

Append this test to `tests/test_patchcore_archive_loading.py`:

```python

def test_patchcore_load_new_archive_sets_score_stats(tmp_path):
    from models import PatchCore

    patch_lib = torch.randn(2, 2, 4, 3)
    stats = {
        "baseline": torch.ones(2, 2),
        "scale": torch.full((2, 2), 2.0),
        "recommended_pixel_threshold": torch.tensor(4.0),
    }
    save_patchcore_archive(str(tmp_path), "new.ts", patch_lib, stats)

    model = PatchCore.__new__(PatchCore)
    result = PatchCore.load(model, str(tmp_path / "new.ts"), [2, 2])

    assert result is True
    assert torch.allclose(model.patch_lib, patch_lib)
    assert torch.allclose(model.score_stats["baseline"], stats["baseline"])
    assert torch.allclose(model.score_stats["scale"], stats["scale"])
```

- [ ] **Step 3: Run the new load tests and verify they fail**

Run:

```bash
python -m pytest tests/test_patchcore_archive_loading.py::test_patchcore_load_old_archive_leaves_score_stats_none tests/test_patchcore_archive_loading.py::test_patchcore_load_new_archive_sets_score_stats -v
```

Expected: FAIL because `PatchCore.load()` still manually reads the first TorchScript parameter and does not set `score_stats`.

- [ ] **Step 4: Update `PatchCore.load()`**

In `indad/models.py`, replace the current `PatchCore.load()` body:

```python
        # Training saves the patch library as a TorchScript archive via
        # `torch.jit.script(...).save(...)`, so we should load it with
        # `torch.jit.load(...)` instead of `torch.load(...)`.
        ts = torch.jit.load(path, map_location="cpu")
        par = ts.named_parameters()
        for key, value in par:
            self.patch_lib = value.detach()
            self.patch_lib.requires_grad_(False)
            break
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        return True
```

with:

```python
        self.patch_lib, self.score_stats = load_patchcore_archive(path)
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        if self.score_stats is None:
            print("score normalization stats not found; using raw PatchCore scores")
        return True
```

- [ ] **Step 5: Add a private helper to choose the inference map**

In `PatchCore`, just above `def predict(...)`, add:

```python
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
```

- [ ] **Step 6: Use normalized map in `PatchCore.predict()`**

In `PatchCore.predict()`, replace:

```python
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        s = s_star.cpu()
```

with:

```python
        raw_map = min_val.view(fmap_h, fmap_w)
        score_map = self._normalize_score_map_if_available(raw_map)

        s_idx = torch.argmax(score_map.reshape(-1))
        s_star = torch.max(score_map)
        s = s_star.cpu()
```

Then replace:

```python
        s_map = min_val.view(1,1, fmap_h, fmap_w)
```

with:

```python
        s_map = score_map.view(1, 1, fmap_h, fmap_w)
```

- [ ] **Step 7: Remove now-unused raw `s_idx` if linting complains**

If no linting is used, leave `s_idx` as-is because the existing code already computed it without using it. If you prefer cleanup, remove this line:

```python
        s_idx = torch.argmax(score_map.reshape(-1))
```

and keep:

```python
        s_star = torch.max(score_map)
```

- [ ] **Step 8: Run tests**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all tests.

- [ ] **Step 9: Commit**

Run:

```bash
git add indad/models.py tests/test_patchcore_archive_loading.py
git commit -m "feat: apply PatchCore position-normalized scores"
```

---

### Task 5: Add raw/normalized diagnostic image saving without changing old fallback behavior

**Files:**
- Modify: `indad/models.py:94-132`, `indad/models.py:620-623`

- [ ] **Step 1: Add a diagnostic save helper**

In `indad/models.py`, after `save_smap_image2(...)`, add:

```python

def save_smap_image_pair(results_dir, img_path, sample, score, raw_s_map, final_s_map, predict_time):
    filename = os.path.basename(img_path).split('.')[0]
    classname = os.path.basename(os.path.dirname(img_path))
    scorename = "{:.2f}".format(score.item())
    predict_time_str = "{:.0f}".format(predict_time * 1000)
    smap_path = os.path.join(
        results_dir,
        classname + '_' + filename + '_' + scorename + '_' + predict_time_str + 'ms_pair.jpg'
    )

    img = tensor_to_img(sample[0], normalize=True)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    raw_map = pred_to_img(raw_s_map).cpu().numpy().squeeze()
    final_map = pred_to_img(final_s_map).cpu().numpy().squeeze()

    raw_heatmap = cv2.applyColorMap((raw_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    final_heatmap = cv2.applyColorMap((final_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    raw_result = cv2.addWeighted(raw_heatmap, 0.5, img, 0.5, 0)
    final_result = cv2.addWeighted(final_heatmap, 0.5, img, 0.5, 0)

    stackimg = cv2.vconcat([img, raw_result, final_result])
    cv2.imwrite(smap_path, stackimg)
```

- [ ] **Step 2: Keep the existing output and add pair output only when normalized stats are used**

In `PatchCore.predict()`, after the final blurred `s_map` is created:

```python
        s_map = s_map.cpu()
        s_map = self.blur(s_map)
```

add raw-map upsample/blur creation:

```python
        raw_s_map = raw_map.view(1, 1, fmap_h, fmap_w)
        raw_s_map = torch.nn.functional.interpolate(
            raw_s_map, size=(height, width), mode='bilinear'
        )
        raw_s_map = raw_s_map.cpu()
        raw_s_map = self.blur(raw_s_map)
```

Then replace the save block:

```python
        # save_smap_image(self.results_dir, path, s, s_map, end_time - start_time)
        save_smap_image2(self.results_dir, path, sample, s, s_map, end_time - start_time)
```

with:

```python
        # save_smap_image(self.results_dir, path, s, s_map, end_time - start_time)
        save_smap_image2(self.results_dir, path, sample, s, s_map, end_time - start_time)
        if self.score_stats is not None and self.match_mode == "exact_position":
            save_smap_image_pair(
                self.results_dir,
                path,
                sample,
                s,
                raw_s_map,
                s_map,
                end_time - start_time,
            )
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all tests.

- [ ] **Step 4: Commit**

Run:

```bash
git add indad/models.py
git commit -m "feat: save PatchCore raw and normalized diagnostics"
```

---

### Task 6: Add a smoke test for score-map normalization helper behavior on a PatchCore shell object

**Files:**
- Modify: `tests/test_patchcore_archive_loading.py`

- [ ] **Step 1: Add direct helper tests for `_normalize_score_map_if_available()`**

Append these tests to `tests/test_patchcore_archive_loading.py`:

```python

def test_patchcore_normalize_score_map_uses_stats_for_exact_position():
    from models import PatchCore

    model = PatchCore.__new__(PatchCore)
    model.score_normalization_enabled = True
    model.score_normalization_clamp_min_zero = True
    model.match_mode = "exact_position"
    model.score_stats = {
        "baseline": torch.tensor([[1.0, 2.0]]),
        "scale": torch.tensor([[1.0, 2.0]]),
    }
    raw_map = torch.tensor([[0.0, 6.0]])

    norm_map = PatchCore._normalize_score_map_if_available(model, raw_map)

    assert torch.allclose(norm_map, torch.tensor([[0.0, 2.0]]))


def test_patchcore_normalize_score_map_returns_raw_for_same_row():
    from models import PatchCore

    model = PatchCore.__new__(PatchCore)
    model.score_normalization_enabled = True
    model.score_normalization_clamp_min_zero = True
    model.match_mode = "same_row"
    model.score_stats = {
        "baseline": torch.tensor([[1.0, 2.0]]),
        "scale": torch.tensor([[1.0, 2.0]]),
    }
    raw_map = torch.tensor([[0.0, 6.0]])

    result = PatchCore._normalize_score_map_if_available(model, raw_map)

    assert result is raw_map
```

- [ ] **Step 2: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_patchcore_archive_loading.py::test_patchcore_normalize_score_map_uses_stats_for_exact_position tests/test_patchcore_archive_loading.py::test_patchcore_normalize_score_map_returns_raw_for_same_row -v
```

Expected: PASS.

- [ ] **Step 3: Run all unit tests**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all tests.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/test_patchcore_archive_loading.py
git commit -m "test: cover PatchCore score map normalization"
```

---

### Task 7: Manual training/prediction verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify import syntax**

Run:

```bash
python -m py_compile indad/patchcore_normalization.py indad/models.py indad/run.py indad/run-yml.py indad/predict.py
```

Expected: command exits with status 0.

- [ ] **Step 2: Run unit tests**

Run:

```bash
python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: PASS for all tests.

- [ ] **Step 3: Run a small real training job if a local dataset is available**

Check for a local dataset path first:

```bash
ls datasets
```

If a usable dataset exists, run one small PatchCore training command using the existing CLI pattern. Example for MVTec `hazelnut`:

```bash
python indad/run.py patchcore --dataset hazelnut --dataset_dir ./datasets --result_dir ./results --image_size 224 --f_coreset 0.1
```

Expected:

- Training prints `score normalization threshold: <number>` when there are at least 4 OK training patches after coreset.
- Result directory contains the final model archive.
- Evaluation completes without crashing.

If no local dataset exists, skip this step and state: `Skipped real training smoke test because no local dataset was available.`

- [ ] **Step 4: Verify old-model fallback manually if an old archive is available**

If an older model archive exists, run prediction using the existing prediction command format. Example:

```bash
python indad/predict.py path/to/old_model.ts --dataset hazelnut --dataset_dir ./datasets --results_dir ./results-predict
```

Expected:

- Console prints `score normalization stats not found; using raw PatchCore scores`.
- Prediction completes using raw score maps.

If no old archive exists, skip this step and state: `Skipped old-model manual fallback because no old archive was available.`

- [ ] **Step 5: Commit only if verification required small fixes**

If Step 1-4 required code changes, commit them:

```bash
git add indad tests
git commit -m "fix: stabilize PatchCore score normalization verification"
```

If no code changes were needed, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage:
  - Leave-one-out nearest-neighbor distances: Task 1.
  - Median/MAD baseline and scale: Task 1.
  - Scale floor and optional smoothing/cap: Task 1.
  - Save/load stats and old-model fallback: Task 2 and Task 4.
  - `exact_position`-only normalization: Task 4 and Task 6.
  - Raw and normalized diagnostic maps: Task 5.
  - Tests and manual verification: Task 1, Task 2, Task 6, Task 7.

- Placeholder scan:
  - No `TBD`, `TODO`, `FIXME`, or unspecified implementation steps are intentionally left in this plan.

- Type consistency:
  - `baseline` and `scale` are always `[H, W]` tensors.
  - `recommended_pixel_threshold` is saved as a scalar tensor and loaded as shape `[]`.
  - `patch_lib` remains `[H, W, N, C]`.
  - `PatchCore.score_stats` is either `None` or a dict with keys `baseline`, `scale`, and `recommended_pixel_threshold`.
