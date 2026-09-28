"""GaugeDefect feature-transport curvature on PatchCore's fused feature lattice.

The raw score belongs to a 2x2 loop, not to an individual feature cell.
Calibration must therefore happen on loops before distributing scores to cells.
"""

import math

import torch
import torch.nn.functional as F


def _orthogonal_transport(source, target):
    """Procrustes transport from [..., C, r] source frames to target frames."""
    cross = source.transpose(-2, -1) @ target
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def loop_curvature(feature_map, rank=8, window=3, chunk_size=256):
    """Return [B, H-1, W-1] uncalibrated loop curvature in float32.

    Frames and transports use chunked batched SVD to bound peak workspace.
    The feature lattice must have at least two cells in either direction.
    """
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    batch, channels, height, width = feature_map.shape
    if height < 2 or width < 2:
        raise ValueError("feature lattice must be at least 2x2")
    if window < 3 or window % 2 != 1:
        raise ValueError("window must be odd and at least 3")
    if rank < 1 or rank > min(channels, window * window - 1):
        raise ValueError("rank exceeds the centered neighborhood dimension")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    # SVD for float16 on CPU is unsupported and is less reliable on GPU.
    features = F.normalize(feature_map.float(), p=2, dim=1, eps=1e-12)
    padded = F.pad(features, (window // 2,) * 4, mode="replicate")
    patches = F.unfold(padded, kernel_size=window)
    # [B, HW, K*K, C]; center samples, then extract right singular vectors.
    patches = patches.transpose(1, 2).reshape(batch * height * width, channels, window * window)
    patches = patches.transpose(1, 2)
    frames = []
    for start in range(0, patches.shape[0], chunk_size):
        neighborhood = patches[start:start + chunk_size]
        neighborhood = neighborhood - neighborhood.mean(dim=1, keepdim=True)
        _, _, vh = torch.linalg.svd(neighborhood, full_matrices=False)
        frames.append(vh[:, :rank, :].transpose(1, 2))
    frames = torch.cat(frames).reshape(batch, height, width, channels, rank)

    def edge_transports(src, dst):
        edge_shape = src.shape[:3]
        src = src.reshape(-1, channels, rank)
        dst = dst.reshape(-1, channels, rank)
        pieces = []
        for start in range(0, src.shape[0], chunk_size):
            pieces.append(_orthogonal_transport(
                src[start:start + chunk_size], dst[start:start + chunk_size]
            ))
        return torch.cat(pieces).reshape(*edge_shape, rank, rank)

    # Only edges that can participate in a complete elementary loop are needed.
    right = edge_transports(frames[:, :, :-1], frames[:, :, 1:])
    down = edge_transports(frames[:, :-1], frames[:, 1:])
    holonomy = (right[:, :-1] @ down[:, :, 1:]
                 @ right[:, 1:].transpose(-2, -1)
                 @ down[:, :, :-1].transpose(-2, -1))
    identity = torch.eye(rank, dtype=holonomy.dtype, device=holonomy.device)
    return torch.linalg.vector_norm(holonomy - identity, dim=(-2, -1)) / math.sqrt(rank)


def loops_to_nodes(loop_scores):
    """Average each loop score over its four incident feature cells."""
    if loop_scores.ndim != 3:
        raise ValueError("loop_scores must have shape [B, H-1, W-1]")
    batch, h, w = loop_scores.shape
    nodes = loop_scores.new_zeros((batch, h + 1, w + 1))
    counts = loop_scores.new_zeros((1, h + 1, w + 1))
    for dy in (0, 1):
        for dx in (0, 1):
            nodes[:, dy:dy + h, dx:dx + w] += loop_scores
            counts[:, dy:dy + h, dx:dx + w] += 1
    return nodes / counts


GAUGE_MODES = {"global": 0, "exact_position": 1, "same_row": 2}


def fit_gauge_stats(curvature_maps, mode, scale_floor=1e-4):
    """Fit normal curvature statistics before any loop-to-node averaging.

    curvature_maps: [N, H-1, W-1]. The three modes respectively aggregate
    across all locations, at each absolute location, or within each row.
    """
    if mode not in GAUGE_MODES:
        raise ValueError("unsupported gauge calibration mode: " + str(mode))
    if curvature_maps.ndim != 3 or curvature_maps.shape[0] < 2:
        raise ValueError("gauge calibration requires at least two normal maps")
    maps = curvature_maps.float()
    if mode == "global":
        values = maps.reshape(-1)
        baseline = values.median()
        scale = (values - baseline).abs().median() * 1.4826
    elif mode == "same_row":
        values = maps.permute(1, 0, 2).reshape(maps.shape[1], -1)
        baseline = values.median(dim=1).values[:, None]
        scale = (values - baseline).abs().median(dim=1).values[:, None] * 1.4826
    else:
        baseline = maps.median(dim=0).values
        scale = (maps - baseline).abs().median(dim=0).values * 1.4826
        # Low-variance locations should not amplify rounding noise.
        positive = scale[scale > scale_floor]
        if positive.numel():
            scale_floor = max(scale_floor, torch.quantile(positive, 0.2).item())
    return {"baseline": baseline, "scale": scale.clamp_min(scale_floor), "mode": mode}


def normalize_gauge(curvature, stats, mode):
    """Positive robust z-score, with explicit calibration-mode/shape checks."""
    if stats["mode"] != mode:
        raise ValueError("gauge statistics were fitted for a different match mode")
    baseline = stats["baseline"].to(device=curvature.device, dtype=torch.float32)
    scale = stats["scale"].to(device=curvature.device, dtype=torch.float32)
    expected = {"global": (), "same_row": (curvature.shape[-2], 1),
                "exact_position": curvature.shape[-2:]}[mode]
    if tuple(baseline.shape) != expected or tuple(scale.shape) != expected:
        raise ValueError("gauge calibration shape does not match the feature lattice")
    return ((curvature.float() - baseline) / scale).clamp_min(0)


def gauge_stats_from_archive(params):
    """Read optional Gauge data; legacy PatchCore archives return None."""
    keys = ("gauge_baseline", "gauge_scale", "gauge_mode", "gauge_rank", "gauge_window")
    present = [key in params for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("incomplete Gauge statistics in PatchCore archive")
    mode_id = int(params["gauge_mode"].item())
    modes = {value: key for key, value in GAUGE_MODES.items()}
    if mode_id not in modes:
        raise ValueError("unknown Gauge calibration mode in archive")
    return {"baseline": params["gauge_baseline"].detach(),
            "scale": params["gauge_scale"].detach(),
            "mode": modes[mode_id], "rank": int(params["gauge_rank"].item()),
            "window": int(params["gauge_window"].item())}
