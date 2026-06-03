# 更新日志

## v4.0

- 新增 PatchCore 按位置分数归一化能力，用于缓解背景区域与纹理/形变区域 raw distance 分布不一致导致的统一阈值失效问题。
- 训练阶段基于 OK 图 patch 特征计算 leave-one-out 最近邻距离，并使用 median/MAD 估计每个位置的 `baseline` 和 `scale`。
- 推理阶段在 `exact_position` 模式下可将原始 PatchCore 距离图转换为位置归一化 z-score 图，旧模型或非 `exact_position` 模式自动回退 raw score。
- 新增 PatchCore 模型归档字段 `score_baseline`、`score_scale`、`recommended_pixel_threshold`，并保持旧版单参数归档兼容。
- `run-yml.py` 入口接入 `score_normalization` 配置，`config/tpl/patchcore-cv2.yml` 增加默认归一化参数。
- 新增 raw / normalized 热力图对比输出，便于观察归一化前后纹理区域和背景区域的分数变化。
- 新增 `indad/patchcore_normalization.py` 单元测试和模型归档兼容性测试。
- 修复 `test.sh`，使用 `run-yml.py` 作为 PatchCore yml 训练入口，并修正 bash 路径和参数换行问题。
- 添加 `POSITION_NORMALIZATION_CHANGES.md`，记录位置归一化设计、配置、测试验证和使用注意事项。

## v3.0

- PatchCore 训练阶段统一按 `exact_position` 方式保存特征库，`patch_lib` 格式固定为 `[H, W, N, C]`。
- PatchCore 推理阶段保留三种匹配模式：`global`、`same_row`、`exact_position`，三种模式均基于统一的按位置特征库派生。
- `global` 模式推理时临时展开全部位置特征，执行全局匹配。
- `same_row` 模式推理时临时按行合并特征，执行同行或行邻域匹配。
- `exact_position` 模式推理时直接使用按位置特征库，执行同位置或位置邻域匹配。
- `patchcore.yml` 的模型参数中新增 `start_pos`、`end_pos`、`match_mode`，便于追踪训练和推理配置。

## v2.1

- 增加 PatchCore 位置邻域搜索支持。
- `exact_position` 匹配模式支持在当前位置周围的空间邻域内查找候选特征。

## v2.0

- 增加 PatchCore `match_mode` 参数，用于控制推理阶段的特征匹配方式。
- 从该版本开始加入 `neighbor_radius` 参数，用于控制邻域搜索半径。
- 增加按行匹配能力。
- `same_row` 匹配模式支持基于 `neighbor_radius` 的行邻域搜索。
- 热力图生成支持不同匹配模式下的异常分数输出。

## v1.0

- 初始标记版本，基于 PatchCore 的工业异常检测实现。
