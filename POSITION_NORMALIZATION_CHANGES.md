# PatchCore 位置归一化改动说明

本文档记录本次围绕 PatchCore 按位置推理新增的改动，以及 `run-yml.py` / `test.sh` 入口相关调整。

## 背景问题

当前 PatchCore 支持按空间位置构建 memory bank：

```text
patch_lib: [H, W, N, C]
```

推理时在 `exact_position` 模式下，每个测试 patch 只和同一位置的正常 patch 做最近邻匹配。

这种方式在背景稳定、零件位置固定时有优势，但存在一个问题：不同位置的正常波动范围不同。

- 背景区域变化小，正常距离分数低。
- 纹理区域、形变区域变化大，正常距离分数天然较高。
- 如果直接对原始最近邻距离使用同一个阈值，背景区域容易区分缺陷，但纹理区域容易误报或漏检。

因此新增了“按位置归一化”的分数处理逻辑。

## 新增功能概览

### 1. Leave-One-Out 正常分布统计

新增模块：

```text
indad/patchcore_normalization.py
```

主要函数：

```python
leave_one_out_nearest_distances(pos_lib)
compute_position_score_stats(patch_lib, ...)
apply_position_normalization(raw_map, baseline, scale, ...)
```

训练阶段会基于 OK 图 patch 特征，计算每个位置的正常距离分布。

对于某个位置 `(h, w)`：

```text
X = patch_lib[h, w] = [N, C]
```

对每个训练 patch，排除自身后找最近邻距离：

```text
loo_dist[i] = min distance(X[i], X[j]), j != i
```

这样可以避免训练图和 memory bank 中自身匹配，导致距离恒为 0。

### 2. 每位置 baseline / scale

对每个位置的 leave-one-out 距离，计算鲁棒统计量：

```text
baseline[h,w] = median(loo_dist)
scale[h,w]    = 1.4826 * median(abs(loo_dist - baseline[h,w]))
```

也就是 median + MAD，而不是 mean + std。

原因：OK 图数量可能较少，且可能存在轻微正常波动或离群样本，median/MAD 更稳健。

### 3. 推理阶段分数归一化

推理时仍然先计算原始 PatchCore 最近邻距离：

```text
raw_map[h,w]
```

如果满足以下条件：

- `score_normalization.enabled = true`
- 当前 `match_mode == "exact_position"`
- 模型归档中包含 score stats
- stats 尺寸与当前 feature map 尺寸一致

则转换为归一化分数：

```text
norm_map[h,w] = max((raw_map[h,w] - baseline[h,w]) / scale[h,w], 0)
```

后续图像级分数和热力图默认使用 `norm_map`。

如果没有 stats，或者不是 `exact_position`，会自动回退到原始 raw score。

## 小样本保护

新增了以下保护策略：

```yaml
score_normalization:
    enabled: true
    min_train_patches: 4
    scale_floor_quantile: 0.20
    scale_cap_quantile: null
    smooth_scale: true
    smooth_kernel: 3
    threshold_quantile: 0.999
    clamp_min_zero: true
```

含义：

- `min_train_patches`: 每个位置至少需要多少 OK patch 才启用统计。
- `scale_floor_quantile`: 给 scale 设置分位数下限，避免背景区域 scale 太小导致过敏。
- `scale_cap_quantile`: 可选 scale 上限，默认关闭。
- `smooth_scale`: 是否对 scale map 做轻微平滑。
- `smooth_kernel`: scale 平滑核大小。
- `threshold_quantile`: 从 OK 图 leave-one-out 分布中估计推荐阈值的分位数。
- `clamp_min_zero`: 归一化后负数分数截断为 0。

## 训练阶段改动

修改文件：

```text
indad/models.py
```

`PatchCore.fit()` 中新增：

1. 在 coreset 采样前保存完整训练 patch 分布：

```python
score_stats_patch_lib = self.patch_lib
```

2. 使用完整 OK patch 分布计算 score stats：

```python
self.score_stats = compute_position_score_stats(score_stats_patch_lib, ...)
```

3. 保存模型时使用新的归档保存函数：

```python
save_patchcore_archive(self.results_dir, 'patch_lib.ts', self.patch_lib, self.score_stats)
```

注意：

- memory bank 仍然可以经过 coreset 压缩。
- baseline/scale 使用 coreset 前的完整 OK 分布计算，这样在 OK 图较少或 coreset 比例较小时，统计更稳定。

## 模型归档改动

新增保存/加载函数：

```python
save_patchcore_archive(...)
load_patchcore_archive(...)
```

新模型归档中包含：

```text
patch_lib
score_baseline
score_scale
recommended_pixel_threshold
```

旧模型归档仍兼容。

如果旧模型只有参数 `0`，加载时会得到：

```python
score_stats = None
```

推理会打印：

```text
score normalization stats not found; using raw PatchCore scores
```

然后使用原始 PatchCore 分数。

## 推理阶段改动

修改文件：

```text
indad/models.py
```

新增方法：

```python
PatchCore._normalize_score_map_if_available(raw_map)
```

该方法负责：

1. 判断是否启用归一化。
2. 判断是否为 `exact_position`。
3. 判断 stats 是否存在。
4. 判断 stats shape 是否匹配。
5. 返回归一化后的 score map 或原始 raw map。

## 诊断图输出

新增函数：

```python
save_smap_image_pair(...)
```

当模型包含 score stats 且使用 `exact_position` 时，会额外保存 raw / normalized 对比图：

```text
原图
raw heatmap
normalized heatmap
```

文件名后缀：

```text
*_pair.jpg
```

这样可以对比：

- 原始 PatchCore 距离图是否在纹理区基线偏高。
- 归一化后是否降低了正常纹理区域的误报。

## run-yml.py 入口改动

修改文件：

```text
indad/run-yml.py
```

实际训练入口是 `run-yml.py`，因此新增从 yml 配置读取：

```python
score_normalization = cfg.get('score_normalization', {})
```

并传入 `PatchCore(...)`：

```python
score_normalization_enabled=score_normalization.get('enabled', True)
score_normalization_min_train_patches=score_normalization.get('min_train_patches', 4)
score_normalization_scale_floor_quantile=score_normalization.get('scale_floor_quantile', 0.2)
score_normalization_scale_cap_quantile=score_normalization.get('scale_cap_quantile', None)
score_normalization_smooth_scale=score_normalization.get('smooth_scale', True)
score_normalization_smooth_kernel=score_normalization.get('smooth_kernel', 3)
score_normalization_threshold_quantile=score_normalization.get('threshold_quantile', 0.999)
score_normalization_clamp_min_zero=score_normalization.get('clamp_min_zero', True)
```

配置模板也已更新：

```text
config/tpl/patchcore-cv2.yml
```

新增默认配置：

```yaml
score_normalization:
    enabled: true
    min_train_patches: 4
    scale_floor_quantile: 0.20
    scale_cap_quantile: null
    smooth_scale: true
    smooth_kernel: 3
    threshold_quantile: 0.999
    clamp_min_zero: true
```

## test.sh 改动

新增/修复脚本：

```text
test.sh
```

当前内容：

```bash
python indad/run-yml.py --cfg_path config/dataset.yml --start_pos 256 --end_pos 2304
```

修复点：

1. 将 Windows 反斜杠路径改为 bash 可识别的正斜杠：

```text
indad\run-yml.py -> indad/run-yml.py
```

2. 将 `--end_pos 2304` 放到同一行，避免 `2304` 被 bash 当成单独命令执行。

## 测试与验证

已在 `transfusion` 环境中验证。

### 单元测试

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

结果：

```text
11 passed
```

### 编译检查

```bash
conda run -n transfusion python -m py_compile indad/models.py indad/run-yml.py
```

结果：通过。

### test.sh 实际运行

使用 UTF-8 输出环境变量运行，避免 Windows GBK 控制台编码问题：

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n transfusion bash test.sh
```

运行结果已到训练和测试结束：

```text
Test results rubberring │ image_rocauc: 1.00 │ pixel_rocauc: -1.00
Results written to ./results\20260601_122548_patchcore_resnet18_2,3_2560x128/patchcore.yml
image_info=3x128x2560
```

## 注意事项

1. 当前归一化只建议用于 `exact_position`。
2. `same_row` 和 `global` 模式会改变最近邻候选分布，不能直接复用 `exact_position` 的 baseline/scale。
3. 如果 OK 图太少，小于 `min_train_patches`，归一化会自动禁用并回退 raw score。
4. 如果使用 `conda run` 时遇到 Windows 编码错误，建议加：

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1
```

5. 当前 `config/dataset.yml` 是实际数据配置文件，本文档没有覆盖其内容。
