# A-MEM 基线与真实数据构建说明

## 本次修改内容

- 实验循环中已加入 `baseline_amem_adapter`，用于把 A-MEM 作为长期记忆 baseline 进行公平对照。
- 数据构建新增真实数据强制模式，抓不到 PMOA-TTS 或 PMC-Patients 时会直接失败，不再静默使用内置 fallback：

```bash
python -m medimem data build --n 50 --output data/processed/samples_50_real.jsonl --require-real-data
```

- 对无法直接访问 Hugging Face 的服务器，支持使用离线真实数据缓存：

```bash
python -m medimem data build \
  --n 50 \
  --output data/processed/samples_50_real.jsonl \
  --require-real-data \
  --cache-dir /home/syh/mem_ehr_hf_cache
```

## 离线缓存格式

缓存目录中可以放以下两个文件：

- `pmoa_tts.json`
- `pmc_patients.json`

文件内容支持两种格式：

- Hugging Face Dataset Viewer `/rows` 接口返回的完整 JSON。
- 直接由数据行对象组成的 JSON 列表。

只要缓存中能读到真实行，`--require-real-data` 就会继续构建数据集；如果 PMOA-TTS 或 PMC-Patients 任一数据源缺失，则命令会失败并输出原因。

## 当前服务器情况

当前服务器直接访问 `datasets-server.huggingface.co` 会报：

```text
Network is unreachable
```

因此在服务器上复现实验时，建议先在有网络的机器预取 PMOA-TTS 和 PMC-Patients 数据行，再上传到 `/home/syh/mem_ehr_hf_cache`，然后使用 `--cache-dir` 构建真实样本。

本次已验证的服务器命令：

```bash
python -m medimem data build \
  --n 50 \
  --output data/processed/samples_50_real.jsonl \
  --require-real-data \
  --cache-dir /home/syh/mem_ehr_hf_cache

python -m medimem data validate --dataset data/processed/samples_50_real.jsonl
```

验证结果：成功构建并校验 50 条真实来源样本，数据备注显示 PMOA-TTS 和 PMC-Patients 均来自 cache，而不是 fallback。

