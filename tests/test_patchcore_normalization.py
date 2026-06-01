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
