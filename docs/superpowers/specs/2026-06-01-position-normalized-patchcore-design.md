# Position-Normalized PatchCore Design

Date: 2026-06-01

## Problem

The current PatchCore implementation can store and query the memory bank by spatial position. Training builds a patch library shaped like `[H, W, N, C]`, and `exact_position` inference compares each test patch only against normal patches from the same feature-map position.

This is useful when product alignment is stable, but raw nearest-neighbor distances are not comparable across positions:

- Background positions have low normal variation, so raw distances are small and a strict threshold can separate defects.
- Textured or deformable part positions have higher normal variation, so raw distances are naturally high even for OK samples.
- A single global threshold over raw distances either over-flags textured regions or misses defects there.

The project currently only has OK images for training, and the OK set can be small enough that no separate calibration split should be held out.

## Goal

Add an unsupervised score-normalization layer for `exact_position` PatchCore inference. The model should continue using the existing position-indexed memory bank, but should convert the raw distance map into a position-normalized anomaly map before thresholding.

The normalized score should answer:

> How abnormal is this patch relative to the normal distance distribution at this same position?

Instead of:

> How large is this raw nearest-neighbor distance globally?

## Non-Goals

This design does not add supervised learning, NG-image calibration, dual memory banks, region clustering, or a new backbone. It also does not change the first implementation of `same_row` or `global` matching. Those modes can continue using raw scores unless a matching-mode-specific normalization is added later.

## Existing Data Flow

Training currently performs the following steps in `PatchCore.fit`:

1. Extract feature maps from OK images.
2. Resize selected feature maps to a common feature-map size.
3. Concatenate feature channels.
4. Convert the patch tensor into `[H, W, 1, C]` per image.
5. Concatenate images into `patch_lib: [H, W, N, C]`.
6. Optionally apply per-position coreset sampling.
7. Save `patch_lib.ts`.

Inference currently performs the following steps in `PatchCore.predict`:

1. Extract and resize test image features.
2. Compute nearest-neighbor distances according to `match_mode`.
3. Reshape distances to `[H, W]`.
4. Upsample to image size.
5. Blur and save the score map.

## Proposed Approach

Add robust per-position score statistics computed from OK training patches by leave-one-out nearest-neighbor distances.

For each position `(h, w)`, let:

```text
X = patch_lib[h, w] = [N, C]
```

For each training patch `X[i]`, compute its nearest-neighbor distance to all other OK patches at the same position:

```text
loo_dist[i] = min distance(X[i], X[j]), j != i
```

This avoids the self-match problem. If `X[i]` were compared against the full memory bank including itself, the nearest distance would be zero, which would underestimate normal variation.

For each position, compute robust statistics:

```text
baseline[h,w] = median(loo_dist)
scale[h,w]    = 1.4826 * median(abs(loo_dist - baseline[h,w]))
```

`median` and `MAD` are used instead of `mean` and `std` because small OK sets can contain mild outliers caused by acceptable variation, acquisition noise, or mislabeled OK images.

At inference time, convert the raw distance map into a normalized map:

```text
norm_map[h,w] = (raw_map[h,w] - baseline[h,w]) / (scale[h,w] + eps)
norm_map[h,w] = max(norm_map[h,w], 0)
```

The normalized map becomes the default map for thresholding and visualization when normalization is enabled and compatible with the match mode.

## Small-Sample Protections

Because the OK set may be small, normalization must include conservative safeguards.

### Minimum Patch Count

Per-position leave-one-out statistics require enough OK patches. If the number of patches at a position is below `min_train_patches`, the implementation should not trust per-position scale estimates. The initial default should be:

```yaml
min_train_patches: 4
```

When the requirement is not met, the model should either disable normalization or fall back to global robust statistics. The first implementation should prefer disabling position normalization with a clear warning, because an unstable normalization can be worse than raw scoring.

### Scale Floor

Stable background positions can produce near-zero MAD values. Without a floor, tiny normal changes can create extreme z-scores and false positives. Apply a data-derived lower bound:

```text
scale_floor = max(absolute_eps, quantile(valid_scale_values, scale_floor_quantile))
scale[h,w] = max(scale[h,w], scale_floor)
```

Recommended default:

```yaml
scale_floor_quantile: 0.2
```

This keeps background sensitivity, but prevents the denominator from becoming unrealistically small.

### Optional Scale Cap

Highly variable texture positions can produce very large scale values. A large scale can suppress true defects in textured areas. Keep scale capping as a configuration option, but do not enable it by default:

```yaml
scale_cap_quantile: null
```

If enabled later, a reasonable starting point is `0.95`.

### Optional Spatial Smoothing

Position-wise scale estimates can be noisy when there are few OK images. Neighboring feature-map positions often share similar normal variation, so a light blur can stabilize the scale map:

```yaml
smooth_scale: true
smooth_kernel: 3
smooth_baseline: false
```

The baseline should not be smoothed by default because material boundaries can create real baseline discontinuities.

## Threshold Recommendation

Without NG images, thresholds can only be calibrated to expected false-positive behavior on OK data.

After computing leave-one-out distances and per-position statistics, normalize the training leave-one-out distances:

```text
loo_z[h,w,i] = (loo_dist[h,w,i] - baseline[h,w]) / scale[h,w]
loo_z = max(loo_z, 0)
```

Then compute a recommended pixel threshold from a high quantile:

```text
recommended_pixel_threshold = quantile(loo_z, threshold_quantile)
```

Initial default:

```yaml
threshold_quantile: 0.999
```

This threshold means approximately: flag pixels whose normalized score is above the 99.9th percentile of OK leave-one-out behavior. It is not guaranteed to be optimal for every defect type because no NG data is available.

## Model Artifacts

Keep the existing `patch_lib.ts` artifact for compatibility.

Add a second artifact for normalization statistics:

```text
patch_stats.ts
```

It should contain at least:

```text
baseline: [H, W]
scale: [H, W]
recommended_pixel_threshold: scalar
normalization_config: serializable metadata
```

If `patch_stats.ts` is absent at inference time, the model must fall back to raw PatchCore scoring and log a warning. Existing trained models should continue to work.

## Inference Behavior

When all of the following are true:

- `score_normalization.enabled` is true,
- `match_mode` is `exact_position`,
- `patch_stats.ts` is available,
- stats shape matches the feature-map size,

then inference should compute both:

```text
raw_map:  [H, W]
norm_map: [H, W]
```

The normalized map should be used as the primary score map. The raw map should remain available for diagnostics.

When any condition is not true, inference should use the raw map exactly as it does today.

## Match Mode Compatibility

### exact_position

Supported in the first implementation. The normalization statistics are computed from same-position leave-one-out distances, matching the inference distribution.

### same_row

Not supported in the first implementation. `same_row` changes the candidate distribution because a patch can match other positions in the row. If normalization is needed later, statistics must be computed with the same `same_row` candidate rule.

### global

Not supported for position normalization. Global matching removes position semantics. If needed later, use a single global baseline and scale instead of `[H, W]` statistics.

## Configuration

Recommended initial configuration:

```yaml
score_normalization:
  enabled: true
  mode: robust_position
  apply_match_modes: ["exact_position"]
  stats: median_mad
  min_train_patches: 4
  scale_floor_quantile: 0.2
  scale_cap_quantile: null
  smooth_scale: true
  smooth_kernel: 3
  smooth_baseline: false
  clamp_min_zero: true
  threshold_quantile: 0.999
  save_raw_map: true
  save_norm_map: true
```

## Diagnostics

Training should optionally save diagnostic images or arrays for:

- `baseline` heatmap,
- `scale` heatmap,
- normalized leave-one-out score distribution,
- recommended threshold value.

Inference should optionally save both raw and normalized heatmaps. This helps identify whether missed defects are caused by raw nearest-neighbor matching or by normalization suppressing high-variance regions.

## Risks and Mitigations

### OK Set Too Small

Reason: Very small OK sets produce unstable leave-one-out distributions. A single unusual OK image can dominate the estimate.

Mitigation: Require `min_train_patches`, use median/MAD, use scale floors, and optionally smooth the scale map. If sample count is too small, fall back to raw scoring instead of applying unreliable normalization.

### Background Over-Sensitivity

Reason: Background positions can have nearly zero normal variation, producing tiny scale values. Small acquisition noise can then become a very large normalized score.

Mitigation: Apply `scale_floor_quantile`, keep an absolute epsilon, optionally clamp extreme z-scores, and use connected-component area filtering after thresholding.

### Texture Defects Suppressed

Reason: Highly variable texture positions can have large scale values. A true defect may be normalized down if it falls within the estimated broad normal variation.

Mitigation: Save raw and normalized maps for comparison, keep optional `scale_cap_quantile`, and later consider a fused score such as `norm_map + lambda * raw_map_normalized` if diagnostics show systematic suppression.

### Old Model Compatibility

Reason: Existing artifacts only contain `patch_lib.ts`.

Mitigation: Treat missing stats as a non-fatal condition and fall back to raw scoring.

### Match-Mode Distribution Mismatch

Reason: Statistics computed for `exact_position` do not match `same_row` or `global` inference distributions.

Mitigation: Enable normalization only for `exact_position` in the first implementation.

### Additional Training Cost

Reason: Leave-one-out requires pairwise distance computation for each position.

Mitigation: Compute stats once after training and save them. Process one position at a time to keep memory bounded. If `N` becomes large, add sampling for stats estimation.

## Testing Plan

1. Unit-test leave-one-out distance calculation on a small synthetic tensor and verify that self-distances are excluded.
2. Unit-test median/MAD baseline and scale calculation, including scale floor behavior.
3. Unit-test inference fallback when `patch_stats.ts` is missing.
4. Unit-test shape validation between stats and feature-map size.
5. Run a small training/inference smoke test and verify that both raw and normalized maps can be produced.
6. Compare raw and normalized heatmaps on OK images to confirm that high-texture normal regions no longer dominate solely because of higher raw baseline distances.

## Acceptance Criteria

- Existing models without stats still run with raw scores.
- New models can save and load per-position normalization stats.
- `exact_position` inference can produce a normalized score map.
- The normalized score map uses per-position baseline and scale with scale-floor protection.
- Recommended thresholds can be computed from OK leave-one-out normalized scores.
- Raw score maps remain available for diagnostics.
