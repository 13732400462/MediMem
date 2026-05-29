# A-MEM 原版评测数据缓存

本目录缓存 A-MEM 原论文/源码使用的公开评测数据，供后续备用。

## LoCoMo

- 来源：https://github.com/snap-research/locomo
- 官方文件：`locomo/locomo10.official.json`
- 扁平 QA：`locomo/locomo10.qa_flat.jsonl`
- 规模：10 个超长 conversation，1,986 条 QA。
- 校验：`locomo10.official.json` 与 `A-mem-main/data/locomo10.json` 的 SHA256 一致。

## DialSim / LongDialQA

- 来源：https://huggingface.co/datasets/jiho283
- 本地文件：`dialsim/*/*.parquet`
- 子集规模：
  - `dialsim-friends`：788 行
  - `dialsim-bigbang`：805 行
  - `dialsim-theoffice`：2,347 行
- 合计：3,940 行 session 级数据。

说明：DialSim 每行内部包含多组问题、选项和答案数组，直接展开会生成非常大的 JSONL。当前先保留官方 parquet 原始分片，后续需要时再按任务格式定向转换。
