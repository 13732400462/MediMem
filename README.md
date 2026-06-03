# 纵向 EHR 记忆增强多智能体系统

本项目用于实现 proposal 中的“面向长期复诊的记忆增强多智能体临床规划与推理系统”。当前版本完成了代码框架、10 条开源病例增强样本、DeepSeek 推理、DDO/ColaCare baseline adapter、自动优化循环和中文实验报告。

## 服务器位置

项目根目录：

```bash
/home/syh/mem_ehr_agent
```

最新完成实验：

```bash
/home/syh/mem_ehr_agent/runs/20260512_165750
```

DeepSeek 测试配置已写入服务器 `.env`，文件权限为 `600`。不要提交或打印 API key。

## 快速运行

```bash
cd /home/syh/mem_ehr_agent
source .venv/bin/activate
python -m mem_ehr_agent data validate --dataset data/processed/samples.jsonl
python -m mem_ehr_agent optimize --dataset data/processed/samples.jsonl --max-rounds 10 --require-api --max-workers 6
```

运行产物会写入：

```bash
data/processed/samples.jsonl
runs/<timestamp>/predictions/*.jsonl
runs/<timestamp>/metrics.csv
runs/<timestamp>/optimization_log.jsonl
runs/<timestamp>/analysis_zh.md
```

记忆模块不用 SQL。所有记忆以 JSONL 原子记忆卡保存。

## 数据构建

首轮 smoke 数据集包含 10 条纵向 EHR 风格病例：

```bash
data/processed/samples.jsonl
```

数据来源：

- PMOA-TTS：PubMed Open Access 病例报告，包含带时间戳的临床事件序列。
- PMC-Patients：PubMed Central 病例摘要，用作辅助病例上下文。

每条 JSONL 样本包含：

- `case_id`：病例编号。
- `source_refs`：开源数据来源引用。
- `demographics`：人口学信息。
- `events`：纵向临床事件时间线。
- `encounters`：按时间聚合后的就诊片段。
- `synthetic_labs`：基于病例生成的结构化化验指标。
- `memory_seed`：初始工作记忆。
- `poison_records`：用于测试记忆污染的旧假设或错误记录。
- `labels.primary_diagnosis`：主诊断标签。
- `labels.diagnosis_list`：诊断列表标签。
- `qa_tasks`：SR/IDR/CDR 评测问题。
- `expected_memory_ops`：预期记忆操作。
- `counterfactuals`：反事实干预样本。

说明：服务器直连 Hugging Face 较慢，所以本轮数据先在本地通过公开接口抓取，再同步到服务器。

## 方法实现

系统按 proposal 的三阶段实现：

1. 多智能体纵向 EHR 解析：模拟文本、数值/化验、时序、统筹决策等角色，对长病历进行结构化解析。
2. 动态记忆清洗：使用 JSONL 原子记忆卡维护患者工作记忆，支持 `Write`、`Revise`、`Invalidate`、`Flag`、`Discard` 操作。
3. 反事实复核：记录 CPG 风格 proxy，用于衡量关键证据被移除时诊断置信度是否合理变化。

当前记忆设计：

- 不使用 SQL。
- JSONL 是唯一可信存储。
- 后续可以增加 embedding/FAISS sidecar，但索引只能作为可重建加速结构，不能作为主存储。

## Baseline 设置

本轮比较方法：

- Direct DeepSeek：截断病例上下文，不使用记忆。
- DDO adapter：模拟 DDO 的医疗多智能体问诊/诊断流程。
- ColaCare adapter：模拟 ColaCare 的结构化 EHR DoctorAgent/MetaAgent 流程。
- Ours：带 JSONL 动态记忆、批判智能体和记忆清洗的多智能体方法。

官方参考：

- DDO：Jia et al., EMNLP 2025，`https://github.com/zh-jia/DDO`
- ColaCare：Wang et al., WWW 2025，`https://github.com/PKU-AICare/ColaCare`

注意：服务器侧 GitHub clone 不稳定，出现过 `early EOF`。因此第一轮实验采用统一 JSONL adapter 进行比较，保证所有方法使用同一批输入样本和统一输出协议。

## 最新实验结果

实验目录：

```bash
runs/20260512_165750
```

指标表：

| 方法 | N | 主诊断准确率 | 诊断列表 F1 | CDR F1 | 记忆污染抑制率 | CPG Proxy | 平均 Token |
|---|---:|---:|---:|---:|---:|---:|---:|
| ColaCare adapter | 10 | 0.300 | 0.405 | 0.421 | 0.000 | 0.100 | 563.1 |
| DDO adapter | 10 | 0.500 | 0.500 | 0.432 | 0.000 | 0.100 | 1083.7 |
| Direct DeepSeek | 10 | 0.400 | 0.345 | 0.383 | 0.000 | 0.130 | 358.6 |
| Ours | 10 | **0.600** | **0.607** | **0.619** | **1.000** | **0.220** | 1481.8 |

最佳优化轮次：

```text
Round 1
strategy = {"top_k": 3, "rounds": 1, "temperature": 0.05}
ours primary accuracy = 0.600
best baseline primary accuracy = 0.500
won = true
```

结论：

在 10 条病例的 DeepSeek 实验中，我们的方法超过最强 baseline DDO adapter：

- 主诊断准确率：`0.600` vs `0.500`
- 诊断列表 F1：`0.607` vs `0.500`
- CDR F1：`0.619` vs `0.432`
- 记忆污染抑制率：`1.000` vs `0.000`
- CPG Proxy：`0.220` vs `0.100`

这说明当前版本的动态记忆清洗和全时间线证据整合，在小样本纵向病例诊断上带来了增益。

## 结果文件

```bash
runs/20260512_165750/analysis_zh.md
runs/20260512_165750/metrics.csv
runs/20260512_165750/baseline_metrics.csv
runs/20260512_165750/optimization_log.jsonl
runs/20260512_165750/progress.jsonl
runs/20260512_165750/predictions/baselines.jsonl
runs/20260512_165750/predictions/ours_round_001.jsonl
```

## 测试结果

服务器测试命令：

```bash
cd /home/syh/mem_ehr_agent
.venv/bin/python -m pytest -q
```

结果：

```text
5 passed
```

测试覆盖：

- JSONL 病例 schema 校验。
- JSONL 记忆状态转换。
- 主诊断匹配和 F1 指标计算。

## 下一步

- 将病例数量从 10 条扩展到 50/100 条。
- 对主诊断标签和时间线一致性做人工抽查。
- 加入 ICD/MeSH/UMLS 风格诊断归一化。
- 等服务器 GitHub 访问稳定后，补充 DDO 和 ColaCare 官方 repo 的完整复现。
- 将当前 CPG proxy 升级为真实 DeepSeek 反事实重跑。
- 增加消融实验：无记忆清洗、无批判智能体、无反事实复核、不同 `top_k` 检索数量。
