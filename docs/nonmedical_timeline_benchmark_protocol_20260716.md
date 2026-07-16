# 非医疗长期时间线第二主实验冻结协议（2026-07-16）

## 目标

在不混入医疗诊断指标的前提下，使用四个非医疗长期时间线数据源评估同一组记忆方法。正文使用一张双层表头大表：第一列表方法，第一层表头为数据源，第二层表头为统一指标。各数据集原生指标、成本和类别分解放入附录。

## 数据源与冻结样本

| 数据源 | 冻结范围 | 时间线特征 |
|---|---:|---|
| LoCoMo | 既有 seed `20260606` 冻结测试 1,000 QA；不得重新抽样 | 多会话双人长期生活事件 |
| LongMemEval-S | 官方 cleaned 版本全部 500 QA | 多会话助手记忆、时间推理、知识更新与拒答 |
| DialSim / LongDialQA | Friends、Big Bang、The Office 三子集按官方带 `*_idxes` 证据映射的 easy QA 家族分层抽样共 1,000 QA；seed `20260716` | 五年多方剧情时间线 |
| RHELM | 官方发布的全部 1,305 QA | 对话、邮件和附件共同演化的个人时间线 |

每个冻结清单必须记录样本 ID、来源文件 SHA-256、抽样 seed、各子集数量和清单 SHA-256。开发集与测试集不得重叠。LoCoMo 继续使用既有 200 题独立开发集选出的 `top-k=32`；正式测试不再调参。为避免逐数据集测试集调参，检索预算 `top-k=32` 在其余三个数据集上也保持冻结。

DialSim 仅纳入 `easy_qs_ans_w_time`、`easy_qs_ans_wo_time`、`easy_qs_before_event_unans`、`easy_qs_dont_know_unans` 与 `easy_qs_dont_know_unans_time`。逐题 `*_idxes` 经官方包与全量数据核验后属于 QA/oracle 索引，并非可直接映射到运行时 Script 时间线条目的来源 ID，因此不得伪装成 Evidence R@5 gold；DialSim 在正文改报其原生重要指标平均响应延迟，官方索引保留在逐题 metadata 供审计。

## 方法

1. Direct
2. Static RAG
3. A-MEM
4. DDO
5. G-Memory
6. MemInsight
7. MemoryOS
8. Mem0
9. Letta/MemGPT
10. MediMem (Ours)

表格方法单元格不增加 `adapter`、`official` 或符号。DDO、G-Memory、MemInsight 的统一协议适配性质只在 caption 或正文表外说明，不把适配结果称为原论文在四数据集上的官方原生复现。
Letta/MemGPT 使用官方 `SyncServer`、`CreateAgent`、`BlockUpdate` 与 `AgentLoop` 适配器：时间线通过有界 core-memory block 写入，超出预算时使用同一冻结模型压缩，QA 仍由 Letta agent 从 core memory 回答。该说明只放在 caption 或正文表外，不改写表内方法名。

## 模型与运行约束

- 生成模型固定为 Qwen3-VL-8B，同一 checkpoint、prompt-visible 字段、温度、最大输出长度、QA prompt 和 evaluator。
- 正式模型只在服务器 `/home/syh/A-mem-aaai` 或经核验内容一致的正式工作副本运行。
- 仅使用 GPU1；端口优先为 `8001`。不得停止、重启、查询负载或复用 GPU0 上的既有服务来完成本实验。
- 必须 `require_api=true`。API 错误、解析错误、无输出、超时和 worker failure 均按失败记录，不生成 fallback 答案。
- 写入记忆的字段只来自运行时可见的时间线内容。问题、参考答案、supporting evidence、judge 标签和 evaluator 字段不得进入记忆构建或 QA prompt。
- 测试温度固定为 0。每个方法在每个数据集执行一次冻结测试；不因测试结果调整参数。置信区间使用相同样本 ID 上的 10,000 次配对 bootstrap。

## 统一主指标

- `Answer Acc. ↑`：使用固定 judge prompt 和固定 evaluator 对语义正确性作二元判断。judge 输入只能包含问题、参考答案和方法输出，不包含方法名称。
- `Evidence R@5 ↑`：前五条实际检索结果命中官方 supporting evidence 的样本级 recall，再对有可映射来源证据标注的适用样本取平均。Direct、Letta core memory 以及 DialSim 无可比的排序来源证据，表中记为 `--`，不得当作 0 参与平均。
- `Latency (ms) ↓`：DialSim 使用端到端单题 QA 延迟，保留其实时模拟器的原生评价重点；不与 Evidence R@5 混算。
- `Average`：四个数据集的 Answer Acc. 算术平均；Evidence R@5 只对具有检索输出的方法按四数据集算术平均。

附录保留 LoCoMo QA F1/BLEU-1、LongMemEval 官方 auto-eval、DialSim time-constrained accuracy/latency、RHELM 类别准确率，以及每种方法的平均 token、调用次数、延迟和失败数。

## 正式验收门禁

每个数据集 × 方法单元格必须同时满足：

- 预测数等于冻结样本数，sample ID 唯一且集合完全一致；
- `blocked=0`、`fallback=0`、`api_error=0`、`worker_failure=0`；
- gold answer、supporting evidence 和 evaluator 字段的运行时泄漏计数为 0；
- MediMem full-context shortcut guard 全部通过；
- manifest、逐题预测、逐题 judge、逐题 retrieval、overall metrics 和类别 metrics 均落盘；
- 失败只重跑缺失 ID，最终合并必须保留原始 provenance，禁止用其他方法输出或直接 LLM 输出补位。

任何单元格未通过门禁时，正文大表对应位置保持为空，不从旧 run 或不同协议结果补数。

## 正文表格结构

第一层表头依次为 LoCoMo、LongMemEval、DialSim、RHELM、Average；LoCoMo、LongMemEval 与 RHELM 的第二层为 `Answer Acc.`、`Evidence R@5`，DialSim 为 `Answer Acc.`、`Latency (ms)`；Average 为四源 `Answer Acc.` 与三个具备来源证据数据集的 `Evidence R@5`。方法在第一列。最佳结果加粗、第二名加下划线；成本列不参与最佳/次优标记。
