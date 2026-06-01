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
