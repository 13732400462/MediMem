# 两因素消融与医疗多骨干模型实施计划

## 目标

在同一冻结医疗协议下完成：

1. PMOA-TTS 5,000 例的 `-E-C`、`-E-K`、`-C-K` 三个两因素消融；
2. Llama-3.1-70B、DeepSeek-V3、GPT-4.1、Gemini-2.5-Flash 上 Direct、CoT、A-MEM、CliCARE、MediMem 五条流程的五源医疗实验；
3. 完整审计、归档和后续论文表格生成。

## 任务 1：扩展并测试消融组合

修改 `mem_ehr_agent/optimizer.py`：

- 注册三个两因素消融名称与精确功能开关；
- 将新方法加入消融指标和差值输出白名单；
- 保持所有现有单因素名称与默认行为不变。

修改 `tests/test_memory.py`：

- 验证三个别名均解析为正确的两个功能开关；
- 验证未知或重复名称仍按现有规则处理；
- 验证每个两因素变体恰好改变两个声明开关。

运行：

```text
pytest tests/test_memory.py tests/test_metrics.py -q
```

## 任务 2：增加精确五流程模式

修改 `mem_ehr_agent/baselines.py` 和 `mem_ehr_agent/optimizer.py`：

- 增加 released-source CliCARE adapter；
- 增加只运行 Direct、CoT、A-MEM、CliCARE 的 `backbone` baseline set；
- 由 `full` 消融组提供 MediMem，组成精确五流程；
- 不运行 Static RAG、DDO、ColaCare 或其他无关方法；
- 在 manifest、pipeline comparison 和 gate 中记录五条目标流程。

修改 `mem_ehr_agent/cli.py` 和测试：

- 暴露 `--baseline-set backbone`；
- 验证该模式输出的方法集合恰好正确；
- 保留 `required` 等历史模式的行为。

运行：

```text
pytest tests/test_memory.py tests/test_optimizer_gate.py -q
```

## 任务 3：安全读取 WLAI 密钥

修改 `mem_ehr_agent/config.py`：

- 支持通过密钥文件路径读取 API key；
- 命令行和进程环境只传密钥文件路径，不传密钥内容；
- manifest、日志和错误信息不得包含密钥。

增加测试，验证：

- 密钥文件读取成功；
- 显式环境变量的兼容行为；
- 缺失文件安全失败；
- 配置对象的字符串表示不泄露密钥。

运行：

```text
pytest tests/test_llm.py tests/test_llm_budget.py -q
```

## 任务 4：实现多骨干模型正式调度器

新增 `scripts/run_medical_backbone_wlai_20260727.py`：

- 固定四个精确模型 ID、五源 5×500 数据文件和五条流程；
- 查询 `/v1/models` 并只保存脱敏摘要与 SHA-256；
- 对每个模型执行最小 live probe；
- 为四个模型创建独立 run root；
- 并行启动四个 `experiment-suite` 子进程；
- 每个子进程使用 `baseline-set backbone` 和 `ablation-groups full`；
- 记录命令、代码提交、数据哈希、模型 ID、并发度、PID、日志和退出状态；
- 不在任何参数、日志或 manifest 中写入密钥；
- 失败时保留现场，不将不完整单元标记完成。

## 任务 5：实现统一审计器

新增 `scripts/audit_medical_backbone_wlai_20260727.py`：

- 核验四个新增模型 × 五条流程 × 2,500 个样本；
- 核验每源恰好 500 个冻结 ID；
- 核验 prediction 唯一、完整且无 fallback/API/worker/progress 错误；
- 运行 leakage audit；
- 从既有归档核验并复用 Qwen3-VL-30B 五条流程；
- 生成按源指标、宽表、审计 JSON 和 LaTeX 草表；
- 未通过单元格输出 `Pending`，禁止拼接。

## 任务 6：实现 GPU 0 消融启动器

新增 `scripts/run_pairwise_pmoa_ablation_gpu0_20260727.sh`：

- 启动前核对 GPU 0、端口、进程归属、模型路径和冻结数据哈希；
- 只在 GPU 0 启动 Qwen3-VL-8B；
- 仅运行三个两因素变体；
- 使用 strict no-fallback、冻结 seed 和原评估器；
- 记录 PID、端口、利用率与完整命令；
- 完成后只停止本启动器创建的 vLLM 进程。

## 任务 7：本地验证与提交

运行：

```text
pytest -q
git diff --check
```

检查完整差异，只提交本次代码、测试、脚本和计划文件。

## 任务 8：服务器部署和正式启动

在 `/home/syh/A-mem-aaai`：

- 重新检查两张 GPU、进程、端口和目标文件；
- 备份将被覆盖的服务器脚本；
- 同步已提交实现；
- 运行四模型 live probe；
- GPU 0 启动两因素消融；
- GPU 1 归属医疗多骨干实验的本地检索工作，远程模型生成并发执行；
- 启动 GPU、吞吐、错误和进度监控。

## 任务 9：完成、归档与论文更新

等待所有正式单元完成并通过审计后：

- 同步原始结果与审计材料；
- 生成第 4.2 节新表和消融补充行；
- 保持第 4.1 节 Table 2 数值不变；
- 将非医疗表移动至附录；
- 编译并逐页检查论文；
- 更新 PDF、Overleaf ZIP、网页端 Overleaf 和两份项目上下文。
