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
