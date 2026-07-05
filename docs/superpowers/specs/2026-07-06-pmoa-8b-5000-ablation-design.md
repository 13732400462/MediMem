# PMOA-TTS 8B 5000 样本消融实验设计

## 目标

在临时 V100 服务器上使用 Qwen3-VL-8B，将论文现有 PMOA-TTS 组件消融从 1000 例扩大到 5000 例，检验扩大样本后 full pipeline 与 w/o memory cleaning 的 PDO、Top-1 和 MPC 关系是否稳定。

## 固定实验口径

- 服务器：`yh_shi@192.168.22.101`，项目限定在 `/home/yh_shi/A-mem`。
- 模型：`/home/models/Qwen3-VL-8B-Instruct`，served name `qwen3-vl-8b`。
- 硬件：优先使用空闲 GPU1；启动前检查三张 GPU、端口和进程归属，不触碰其他用户任务。
- 数据：PMOA-TTS 真实数据 5000 例，固定随机种子 `20260606`，禁止 fallback。
- 四个组使用完全相同且顺序一致的病例集合：`full`、`no_memory_cleaning`、`no_evidence_note_injection`、`ablate_with_polluted_memory`。
- 不运行 baseline，使用 `--baseline-set none`；目标总进度为 20000 个 method--case。
- 推理沿用 fast-formal、risk-sample counterfactual、strict no-leak 和既有指标实现，不调整 prompt、top-k、温度或评测公式。

## 数据构建与去重门禁

优先从既有真实数据缓存构建 5000 例 PMOA-TTS 固定切片。临时服务器若缺少完整缓存，则从旧服务器只读迁移所需缓存或已构建数据，并记录源文件与目标文件 SHA-256；不得重新生成 fallback 病例。

正式运行前必须同时满足：

1. JSONL 恰好包含 5000 条记录；
2. 5000 个 case ID 全部唯一；
3. 对排除纯标识字段后的 runtime-visible 病例内容做稳定规范化并计算 SHA-256，5000 个内容哈希全部唯一；
4. 数据校验通过，`source_real_false=0`，无构建期泄漏发现；
5. 四个消融组读取同一个不可变数据文件，禁止各组分别抽样。

若唯一内容少于 5000，实验停止并报告真实可用上限，不通过复制、改 ID 或重复采样凑数。

## 运行流程

1. 只读检查 GPU、端口、进程树、TCP 客户端、仓库提交和工作树状态。
2. 准备并校验 5000 条无重复 PMOA-TTS 数据，保存数据哈希和去重审计结果。
3. 在空闲 GPU1 启动 V100 兼容的 FP16 vLLM：端口 8001、`max-model-len=8192`、`gpu-memory-utilization=0.85`、eager、`max-num-seqs=8`，使用 Triton attention。
4. 对同一数据执行 2 例四组 strict smoke；确认 8/8 成功且无 fallback、API error 和 critical leakage。
5. 以 workers 8 启动正式四组消融，持续记录 launcher、vLLM PID、日志和 run directory。
6. 完成后核验进度、方法计数、case ID 集合、审计、预测错误和最终指标；验收通过后仅精准关闭本项目 vLLM。

## 验收与结果解释

正式完成必须满足：

- `progress.jsonl=20000/20000` 且全部 `ok=true`；
- 四组各 `n=5000`，每组 case ID 集合完全相同；
- failed、blocked、fallback、API error、critical leakage 和 needs-review 均为 0；
- 输出 `metrics.csv`、`source_metrics.csv`、`leakage_audit.jsonl`、gate、预测文件和运行 provenance；
- 汇报四组 PDO、Top-1、MPC 及 full 与 no-memory-cleaning 的差值和适当的不确定性分析。

样本扩大后若 no-memory-cleaning 仍有更高 PDO/Top-1，但 MPC 为 0，则如实解释为诊断指标与污染控制目标之间的权衡，不能为了符合预期而修改数据、指标或筛选结果。若 full 反超，则报告反超幅度，并检查该结论是否超过抽样波动。
