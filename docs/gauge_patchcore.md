# Gauge curvature diagnostics for PatchCore

This is an optional implementation of feature-transport curvature on the **same pooled,
resized and concatenated features** used by this repository's PatchCore. It follows
the local-frame → Procrustes transport → 2x2 loop sequence of GaugeDefect. It is
an adaptation, not a reproduction of the paper's DINOv2 evaluation. Calibration
uses median/MAD on normal images, whereas the paper reports mean/std.

## Fit

Set `gauge.enabled: true` in `config/tpl/patchcore-cv2.yml` (or your job's YAML),
then train a fresh model. The `.ts` archive includes the curvature baseline,
scale, matching mode, rank and neighborhood size. At least two normal training
images are needed. Existing archives still load with Gauge switched off.

Set `gauge.match_mode` according to the intended matching mode:

| PatchCore matching | Curvature calibration | Assumption |
| --- | --- | --- |
| `exact_position` | median/MAD at each loop location | aligned parts |
| `same_row` | median/MAD within each feature row | vertical placement is stable |
| `global` | median/MAD across all loops | no reliable spatial alignment |

Use the **same match mode** when training and inspecting. The system raises an
error on a mode or feature-grid mismatch rather than silently applying the wrong
normal baseline. The match mode controls Gauge calibration; the memory bank and
coreset format are unchanged.

## Inspect at inference

```bash
python indad/patchcore_predict_simple.py \
  --model path/to/patch_lib.ts --input path/to/test \
  --match-mode exact_position --gauge-diagnostics \
  --output results-gauge
```

Each image writes a numbered `_gauge.npz` with `patchcore_raw`,
`patchcore_score`, `gauge_loop` and `gauge_score`. For tiled images these arrays
have a leading tile dimension. `gauge_loop` has one fewer row and column than
the other maps; each calibrated loop contributes to its four incident cells.
PatchCore's anomaly score, output heatmap and thresholds are unchanged. Gauge
is computed only with `--gauge-diagnostics`; the default inference path incurs
no Gauge SVD cost. The Python API exposes the same arrays as
`predictor.last_diagnostics` after prediction when `gauge_diagnostics=True`.

Before choosing a fusion rule, inspect representative OK parts, knife marks,
scratches and water/oil stains separately; compare defect recall, false alarms,
peak CPU/GPU memory and per-image latency. For unaligned parts, check normal
structural edges particularly carefully. Only deploy a fused score after these
measurements and a threshold recalibration on held-out production images.

Paper: https://arxiv.org/abs/2609.13282
