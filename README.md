# MediMem

MediMem 是一个面向长期复诊场景的记忆增强多智能体临床规划与推理系统。当前实现聚焦推理阶段的可审计记忆治理：构建纵向患者记忆、由 Critic 审查并受控更新、生成与来源对齐的证据笔记，以及执行证据约束诊断和扰动式反事实审计。

本仓库保存可复现的源码、测试、运行脚本和少量经过筛选的公开基准资产。论文源文件、私有数据、API 密钥、完整实验输出和模型权重不属于本代码仓库。

## 核心实现

- `mem_ehr_agent/agents.py`：多智能体推理和诊断流程。
- `mem_ehr_agent/memory.py`：记忆存储、Critic 审查和安全更新。
- `mem_ehr_agent/benchmark.py`：长期记忆 benchmark、检索、基线适配和反事实复核。
- `mem_ehr_agent/metrics.py`：诊断、证据检索和审计指标。
- `mem_ehr_agent/optimizer.py`：实验编排与准入检查。
- `mem_ehr_agent/data_builder.py`、`data_sources.py`、`expanded_data.py`：数据构建与来源适配。
- `scripts/`：正式实验、消融、筛选和审计入口。
- `tests/`：核心记忆、推理、指标、恢复和数据流程测试。

MediMem 不是新训练的端到端模型。仓库中的“模型”是推理与记忆治理代码；基础 LLM、embedding 和 reranker 权重由运行环境单独提供，不提交 Git。

## 环境安装

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如需语义检索依赖：

```bash
python -m pip install -e ".[dev,semantic]"
```

## 配置与运行

以 `config/example.env` 为模板，在本地创建 `.env` 或通过进程环境变量传入配置。不要把真实密钥写入代码、脚本参数、日志、实验 manifest 或 Git。

```bash
python -m mem_ehr_agent --help
python -m pytest -q
```

正式实验需要显式启用真实数据和在线 API 准入检查，并使用冻结的样本 manifest、模型标识、prompt、evaluator 和指标定义。不得将 fallback、失败或不完整 run 写成正式结果。

## 数据、模型与输出边界

以下内容只保留在本地或服务器，不上传 Git：

- `.env`、密钥文件和机器本地配置；
- `data/raw/`、`data/processed/` 和未获再分发许可的数据；
- `runs/`、大规模 predictions、judge 输出、缓存和临时报告；
- `.codex_tmp/`、`models/`、`checkpoints/` 及 `*.safetensors`、`*.bin`、`*.onnx` 等模型文件；
- 论文 LaTeX 源文件、编译产物和 Overleaf 归档。

`datasets/amem_original/` 中当前已跟踪的文件是用于可复现性核验的公开基准资产；新增数据在确认许可、去标识化和体积后才能提交。`outputs/nonmedical_timeline_main_table_20260719.csv` 是经过筛选的聚合结果，不包含逐样本预测或私有输入。

## 研究边界

- 规则、guard、记忆清理和扰动审计是推理期治理逻辑，不应描述为可学习模块或训练目标。
- released-source clinical adapters 通过统一协议接入，不代表完整复现原论文 benchmark。
- 所有论文数值必须回溯到同一正式 run 的 predictions、metrics、manifest、validation 和审计文件。
- 非医疗长期记忆 benchmark 的结果不应被外推为所有场景中的普遍领先。

## 许可证与第三方代码

`A-mem-main/` 是保留许可证和来源说明的第三方参考实现。使用或再分发时分别遵循其目录内的许可证。项目自身许可证尚未单独声明；对外发布前应补充与依赖兼容的许可证。
