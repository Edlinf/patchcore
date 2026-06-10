# 更新日志

## v5.2

- standalone predictor 新增 `--device auto|cuda|cpu` 参数，可显式选择 CPU 或 GPU 推理。
- standalone predictor 新增 `--start-pos` / `--end-pos` 参数，普通小图推理时在 feature patch 宽度维按 `start_pos/8:end_pos/8` 裁剪，并同步裁剪可视化原图区域；大图模式暂不支持该参数组合。
- standalone predictor 新增 `--vis-scale` 可视化缩放参数，并将可视化流程改为先缩小 score map 再归一化，降低大图热力图生成耗时。
- standalone predictor 新增 `--visual` 开关；默认不保存可视化图片，仅输出 `scores.csv`、`metrics.json` 和大图模式下的 `tile_scores.csv`，需要查看图像时显式传入 `--visual`。

## v5.1

- 修复 standalone predictor 从模型文件名解析 `image_size` 时的宽高顺序，按 `3xH xW` 解析为 `[W, H]`，保证与 `Cv2AdaptiveResize` 的输入语义一致。
- standalone predictor 输出图调整为 `[原图 | 叠加图]`，替代 `[heatmap | 叠加图]`，便于直接对照原始外观。
- 新增大图推理模式 `--big-image`，按固定几何参数将拼接大图在内存中切成 tile，逐 tile 计算并拼回整张大图 heatmap，边框和间隔区域保持异常分数 0。
- 大图推理输出 `tile_scores.csv`，记录每个 tile 的行列、坐标和异常分数。
- standalone predictor 的 raw map 计算改为 batch-aware，输入 patch `[B, C, H, W]` 后统一输出 `[B, H, W]`。
- 大图推理将所有 tile 组成 `[B, C, H, W]` 后批量推理，减少重复 backbone 调用。
- `global`、`same_row`、`exact_position` 三种匹配模式均支持 batch raw map；`same_row` 和 `exact_position` 的邻域匹配继续使用向量化滑动窗口。
- `PatchCore` 默认 coreset 数量不再受 `60000//(H*W)` 上限限制，按 `f_coreset * N` 计算；评估阶段当前临时返回 `-1, -1`。

## v5.0

- 新增独立 PatchCore 推理脚本 `indad/patchcore_predict_simple.py`，包含单独的 `PatchCorePredictor` 类，不依赖训练侧 `models.PatchCore`。
- 推理脚本支持单张图片和目录批量推理，输出 `scores.csv`、`metrics.json` 以及合并后的热力图/叠加图。
- 批量推理支持根据父目录名自动推断标签，`good/OK/normal` 记为 0，`bad/NG/defect/abnormal` 记为 1，并计算 image-level ROC AUC。
- 支持加载旧版单参数模型归档和新版包含 `score_baseline`、`score_scale`、`recommended_pixel_threshold` 的归档。
- 使用本地 `hub/checkpoints` 下的 backbone 权重，通过 `torch.hub.set_dir(project_root / "hub")` 避免联网下载。
- 推理脚本新增 `--match-mode`，支持 `global`、`same_row`、`exact_position` 三种匹配模式。
- `same_row` 和 `exact_position` 支持 `--neighbor-radius > 0`，并使用 `pad + unfold` 滑动窗口方式向量化计算邻域候选，替代慢速 Python 循环。
- 默认 `neighbor_radius=0`，默认 `match_mode=exact_position`；归一化 score stats 仅在 `exact_position` 模式下应用。
- 新增 `tests/test_patchcore_predict_simple.py`，覆盖模型归档加载、文件名解析、标签推断、三种匹配模式和邻域半径行为。
- 数据集拆图预处理已迁出本仓库，独立到 `D:/python_project/028-grid-dataset-preprocess`；本仓库训练流程继续使用预处理后的小图数据集。

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
