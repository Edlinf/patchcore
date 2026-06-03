import torch
import torch.nn.functional as F


def leave_one_out_nearest_distances(pos_lib: torch.Tensor) -> torch.Tensor:
    """Return each patch's nearest-neighbor distance excluding itself.

    Args:
        pos_lib: Tensor shaped [N, C] for one spatial position.

    Returns:
        Tensor shaped [N].
    """
    # pos_lib 是某一个固定空间位置上的所有 OK 图 patch 特征。
    # 如果直接和完整 memory bank 比，patch 会匹配到自己，距离变成 0。
    # leave-one-out 的核心就是：计算每个 patch 到“其它 OK patch”的最近距离。
    if pos_lib.ndim != 2:
        raise ValueError(f"pos_lib must be [N, C], got shape {tuple(pos_lib.shape)}")
    if pos_lib.shape[0] < 2:
        raise ValueError("leave-one-out distances need at least 2 patches")

    values = pos_lib.float()
    # 两两计算同一位置 OK patch 之间的特征距离，得到 [N, N] 距离矩阵。
    distances = torch.cdist(values, values)
    # 对角线是 patch 和自身的距离，必然为 0；将其置为 inf 来排除自身匹配。
    diag = torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    distances = distances.masked_fill(diag, float("inf"))
    # 每一行取最小值，即该 patch 到其它 OK patch 的最近邻距离。
    return torch.min(distances, dim=1).values


def _median_mad(values: torch.Tensor):
    # 使用 median 作为该位置正常距离的基线，比 mean 更不容易被少量离群 OK 样本拉偏。
    baseline = torch.median(values)
    # MAD 表示距离围绕 median 的典型波动范围，同样比标准差更鲁棒。
    mad = torch.median(torch.abs(values - baseline))
    # 1.4826 是常用修正系数，使 MAD 在近似正态分布下与标准差尺度接近。
    scale = 1.4826 * mad
    return baseline, scale


def _smooth_2d_map(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # OK 图较少时，每个位置单独估计出来的 scale 可能有噪声。
    # 对 scale map 做轻微空间平滑，可以让相邻位置的正常波动估计更稳定。
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
    # 目标：为 feature map 上每个空间位置单独估计正常分数分布。
    # 背景区域正常波动小，纹理/形变区域正常波动大；直接用 raw distance 做统一阈值不公平。
    # 这里计算 baseline/scale 后，推理阶段可以把 raw distance 转成位置归一化 z-score。
    if patch_lib.ndim != 4:
        raise ValueError(f"patch_lib must be [H, W, N, C], got shape {tuple(patch_lib.shape)}")

    H, W, N, _ = patch_lib.shape
    if N < min_train_patches:
        raise ValueError(
            f"position normalization needs at least {min_train_patches} training patches, got {N}"
        )

    # baseline[h,w]：该位置 OK patch 最近邻距离的典型值。
    # scale[h,w]：该位置 OK patch 最近邻距离的正常波动范围。
    # loo_distances[h,w,i]：第 i 张 OK 图在该位置的 leave-one-out 最近邻距离。
    baseline = torch.empty(H, W, dtype=torch.float32)
    scale = torch.empty(H, W, dtype=torch.float32)
    loo_distances = torch.empty(H, W, N, dtype=torch.float32)

    for h in range(H):
        for w in range(W):
            # 对每个空间位置独立计算，不把其它位置的 patch 混进来。
            # 这样背景、边缘、纹理等不同区域会有各自的正常分数基线。
            distances = leave_one_out_nearest_distances(patch_lib[h, w].float()).cpu()
            loo_distances[h, w] = distances
            baseline[h, w], scale[h, w] = _median_mad(distances)

    valid_scale = scale[torch.isfinite(scale)]
    if valid_scale.numel() == 0:
        scale_floor = torch.tensor(float(absolute_eps), dtype=torch.float32)
    else:
        # scale_floor 用所有位置 scale 的分位数作为下限，避免稳定背景区域 scale 太小。
        # 如果 scale 太小，轻微光照/噪声就会被 z-score 放大成很高的异常分数。
        scale_floor = torch.quantile(valid_scale, scale_floor_quantile)
        scale_floor = torch.maximum(scale_floor, torch.tensor(float(absolute_eps), dtype=torch.float32))

    scale = torch.maximum(scale, scale_floor)

    if scale_cap_quantile is not None:
        # 可选上限：防止高纹理区域 scale 过大，把真实缺陷也压得太低。
        # 默认关闭，因为上限过强会让纹理区域误报增多。
        scale_cap = torch.quantile(scale, scale_cap_quantile)
        scale = torch.minimum(scale, scale_cap)
        scale = torch.maximum(scale, scale_floor)

    if smooth_scale:
        # 平滑后再次应用下限，避免平滑把某些位置 scale 拉回过小。
        scale = _smooth_2d_map(scale, smooth_kernel)
        scale = torch.maximum(scale, scale_floor)

    # 把训练 OK 图的 leave-one-out raw distance 也转换成 z-score。
    # 这个分布用于估计推荐阈值：例如 P99.9 表示 OK 图上约 0.1% 像素会超过该阈值。
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

    # 推理时的 raw_map 是 PatchCore 最近邻距离图。
    # 减去该位置正常 baseline，再除以该位置正常波动 scale，得到跨位置更可比的分数。
    norm_map = (raw_map.float() - baseline.float()) / scale.float()
    if clamp_min_zero:
        # 比正常 baseline 更低的距离不需要作为“负异常”展示，直接截断为 0。
        norm_map = torch.clamp_min(norm_map, 0.0)
    return norm_map
