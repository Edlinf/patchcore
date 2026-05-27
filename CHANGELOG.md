# 更新日志

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
