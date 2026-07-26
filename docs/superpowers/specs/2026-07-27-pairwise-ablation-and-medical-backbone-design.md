# 两因素消融与医疗多骨干模型实验设计

## 一、目标

在不修改当前正文医疗主表既有结果、也不修改已归档非医疗长时间线表格结果的前提下，新增两组实验：

1. 在 PMOA-TTS 组件分析中，补充正文已有三个正向组件的两两联合消融；
2. 在论文第 4.2 节新增医疗多骨干模型比较，覆盖五条推理流程和五个骨干模型。

现有 Qwen3-VL 不同规模比较继续作为第 4.1 节的 Table 2。LoCoMo、LongMemEval、DialSim 和 RHELM 四个非医疗数据源的表格保留已验收数值并移至附录，不重新运行，也不与新增医疗多骨干模型表合并。

## 二、冻结实验矩阵

### 2.1 两因素组件消融

消融实验继续使用现有完整消融相同的 5,000 个冻结 PMOA-TTS 样本、Qwen3-VL-8B、提示词、解码配置、评估器、请求随机种子和严格零回退门槛。

新增三个变体：

- 同时移除 source-aligned evidence notes 和 applied memory cleaning（`-E-C`）；
- 同时移除 source-aligned evidence notes，并将动态检索改为 fixed top-k（`-E-K`）；
- 同时移除 applied memory cleaning，并将动态检索改为 fixed top-k（`-C-K`）。

其中，`E` 表示 source-aligned evidence notes，`C` 表示 applied memory cleaning，`K` 表示 dynamic top-k retrieval。Full MediMem 和已有三个单因素消融行继续作为比较基准。

实现时必须复用现有功能开关的组合，不得为两因素行更换提示词、评估器、样本子集或计分规则。

### 2.2 医疗多骨干模型比较

新增第 4.2 节表格使用现有 Qwen3-VL 尺度主表对应的冻结五源协议，每个数据源 500 个样本：

- MedMCQA：500；
- MedQA：500；
- Medical Meadow WikiDoc：500；
- PMOA-TTS：500；
- PMC-Patients：500。

五条推理流程为：

- Direct；
- CoT；
- A-MEM；
- CliCARE；
- MediMem（Ours）。

五个骨干模型为：

- Qwen3-VL-30B；
- Llama-3.1-70B（`llama-3.1-70b`）；
- DeepSeek-V3（`deepseek-v3`）；
- GPT-4.1（`gpt-4.1`）；
- Gemini-2.5-Flash（`gemini-2.5-flash`）。

Qwen3-VL-30B 的单元格复用其原始完整冻结运行的已验收结果，但在使用前必须重新核验 manifest 和样本 ID，不重新运行该模型。

其余四个远程模型分别运行全部五条推理流程和全部 2,500 个冻结样本，共形成 20 个新增“模型—流程”实验单元，以及 50,000 个“流程—样本”推理单元。

每个实验单元必须使用相同的可见输入、方法适配器、对应提示词模板、最大生成预算、请求随机种子、评估器、指标实现和严格零回退策略。禁止拼接不同运行或不同样本子集的结果。

## 三、API 与模型审计

WLAI 的 OpenAI-compatible 接口为 `https://api.wlai.vip/v1`。API 密钥只能由实验进程从 `/root/.config/medimem/wlai.key` 内部读取，不得出现在命令、日志、manifest、prediction、Git 或论文文件中。

正式运行前，调度器必须记录：

- `/v1/models` 查询的 UTC 时间，以及脱敏响应的 SHA-256；
- 对每个精确目标模型 ID 的最小 live chat-completion probe；
- API 返回的模型身份和非空 usage 字段；
- 冻结样本 manifest 的哈希；
- 代码提交号、提示词与配置哈希、评估器版本和方法适配器身份；
- 仅包含 base URL、不包含密钥的脱敏 API 配置。

`/v1/models` 返回的模型目录可能不完整。因此，判断目标模型是否可用时，以精确模型 ID 的 live probe 是否成功为准，不能仅依据模型是否出现在列表中。

## 四、执行架构

### 4.1 GPU 0：两因素消融

GPU 0 专用于 PMOA-TTS 两因素消融。该卡启动本地 Qwen3-VL-8B 服务，并运行三个两因素变体。通过真实推理 worker 并发尽量提高服务利用率。

### 4.2 GPU 1：医疗多骨干模型实验

GPU 1 专用于医疗多骨干模型实验。目标大模型的生成由远程 API 完成；GPU 1 承担 A-MEM 和 MediMem 适配器已有的 CUDA embedding 与检索计算。Direct、CoT 和 released-source CliCARE adapter 保持各自冻结执行路径。

医疗 Top-1 和 Diagnosis F1 由确定性指标程序计算，不额外引入 LLM judge。不得添加与实验无关的计算来人为制造 GPU-Util。

医疗实验调度器应交错执行不同模型和数据源的任务队列，避免某一队列等待 API 时导致全部 worker 空闲。并发度先通过有限 probe 测试，再在无 rate limit 和传输错误的前提下逐步提高。

每个目标模型必须拥有独立的 run root、进度日志、prediction 文件、metrics 和审计报告。只有在冻结配置逐字节一致时，才允许补跑缺失样本 ID。

GPU 利用率、GPU 显存、API 并发度和端到端吞吐必须分别记录。由于目标模型生成发生在远程 API，GPU 1 持续达到 100% 利用率不作为实验有效性门槛，也无法保证；优化目标是最大化真实有效的端到端吞吐，而不是制造合成负载。

## 五、正文表格结构

现有第 4.1 节 Table 2 继续作为独立的 Qwen3-VL 尺度比较，不修改已有数值。

第 4.2 节新增表格，列结构为：

`Paradigm | Method | Reference | Backbone | MedMCQA | MedQA | WikiDoc | PMOA-TTS | PMC-Patients`

每个数据集单元格报告 `Top-1 / Diagnosis F1`。

行先按 Direct、CoT、A-MEM、CliCARE、MediMem 五条流程分组，再在每组中列出五个骨干模型，与用户确认的参考表结构一致。同一个骨干模型下，每个数据集指标的最佳流程使用粗体，次佳流程使用下划线。

每个单元格报告一次完整冻结运行的点估计，不报告 mean 或 standard deviation。

附录中的非医疗表保留原有已验收数值和协议说明。移动表格时必须同步检查编号与交叉引用，不能造成静默错引。

## 六、验证与结果准入门槛

每个新增正式实验单元必须同时满足：

- 样本 ID 与冻结集合完全一致，数量正确；
- prediction ID 唯一，指标覆盖完整；
- 开启 live API 和真实数据要求；
- fallback、blocked method、API error、worker failure、missing prediction 和 progress failure 均为 0；
- critical leakage 和 needs review 均为 0；
- manifest guard 和 validation guard 全部通过；
- 记录精确目标模型 ID 和方法适配器身份；
- 推理阶段不可见 reference diagnosis、gold answer、evaluator 字段或 adversarial target。

两因素消融还必须验证：每个变体相对 Full MediMem 恰好只改变声明的两个功能开关。

医疗多骨干模型审计必须验证完整的 `4 个新增模型 × 5 条流程` 网格，并单独核验五个复用的 Qwen3-VL-30B 实验单元的来源。

失败或未完成的单元只能作为失败运行归档，不得写入论文。任何未通过全部门槛的数值单元格必须保持为 `Pending`。

## 七、论文与实验归档

全部实验通过验收后：

1. 从审计通过的 metrics 生成两因素消融行；
2. 从审计通过的医疗实验网格生成第 4.2 节新表；
3. 保持第 4.1 节 Table 2 的已有数值不变；
4. 将已验收非医疗表格及其分析移动至附录；
5. 只根据真实完成结果更新正文结论，包括负向或混合结果；
6. 在本地归档 manifest、prediction、metrics、validation、probe、脱敏服务记录、精确命令和利用率轨迹；
7. 按既有流程完成 LaTeX 编译、逐页视觉检查、PDF、Overleaf ZIP 和网页端 Overleaf 同步。

禁止推断、虚构实验数值，也禁止从不同模型、数据版本、样本量、评估器或运行中复制结果。
