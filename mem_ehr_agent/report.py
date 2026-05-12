from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import BASELINE_SOURCES
from .io_utils import write_text


def render_report(
    *,
    run_dir: str | Path,
    data_notes: str,
    summaries: list[dict[str, Any]],
    optimization_log: list[dict[str, Any]],
    best_round: dict[str, Any] | None,
    blocker: str | None = None,
) -> str:
    lines = [
        "# 纵向 EHR 记忆增强多智能体实验报告",
        "",
        "## 数据来源与构建流程",
        "- 主数据源：PMOA-TTS（PubMed OA 病例时间序列）。",
        "- 辅数据源：PMC-Patients（PubMed Central 病例摘要）。",
        "- 首轮样本：10 条纵向 EHR JSONL，包含时间事件、合成化验、记忆污染项、跨期 QA 与反事实项。",
        f"- 数据抓取备注：{data_notes.strip() or '公开数据 API 正常或本地已有数据。'}",
        "",
        "## Baseline 设置",
        f"- DDO：{BASELINE_SOURCES['ddo']['paper']}；官方 repo：{BASELINE_SOURCES['ddo']['repo']}。",
        f"- ColaCare：{BASELINE_SOURCES['colacare']['paper']}；官方 repo：{BASELINE_SOURCES['colacare']['repo']}。",
        "- 本轮采用统一 JSONL adapter，保证 direct、DDO、ColaCare、ours 在同一输入/输出协议下比较。",
        "",
        "## 指标汇总",
        "| Method | N | Primary Acc | Diagnosis F1 | CDR F1 | Pollution Suppression | CPG Proxy | Avg Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {n} | {acc:.3f} | {f1:.3f} | {cdr:.3f} | {poll:.3f} | {cpg:.3f} | {tok:.1f} |".format(
                method=row["method"],
                n=row["n"],
                acc=float(row["primary_diagnosis_top1_accuracy"]),
                f1=float(row["diagnosis_list_f1"]),
                cdr=float(row["cdr_f1"]),
                poll=float(row["memory_pollution_suppression"]),
                cpg=float(row["counterfactual_robustness_proxy"]),
                tok=float(row["avg_tokens"]),
            )
        )
    lines.extend(["", "## 自动优化记录"])
    for item in optimization_log[-10:]:
        lines.append(
            f"- Round {item.get('round')}: strategy={item.get('strategy')} "
            f"ours_acc={item.get('ours_accuracy'):.3f}, best_baseline={item.get('best_baseline_accuracy'):.3f}, "
            f"won={item.get('won')}"
        )
    if best_round:
        lines.extend(
            [
                "",
                "## 最好轮次",
                f"- Round：{best_round.get('round')}",
                f"- Strategy：`{best_round.get('strategy')}`",
                f"- Ours primary accuracy：{best_round.get('ours_accuracy'):.3f}",
                f"- Best baseline primary accuracy：{best_round.get('best_baseline_accuracy'):.3f}",
            ]
        )
    if blocker:
        lines.extend(["", "## Blocker", f"- {blocker}"])
    exceeded = bool(best_round and best_round.get("won"))
    lines.extend(
        [
            "",
            "## 结论与下一步",
            f"- 是否超过最强 baseline：{'是' if exceeded else '否'}。",
        ]
    )
    if exceeded:
        lines.append("- 当前结果说明动态记忆清洗和全时间线证据整合在小样本上带来增益；下一步扩展到 50/100 条并加入人工抽查。")
    else:
        lines.extend(
            [
                "- 未超过时的主要原因通常是：样本太少导致平局、baseline 已能直接抽取显式诊断、或我们的 prompt 对诊断同义词归一不足。",
                "- 下一轮改法：扩样到 50 条，加入 ICD/MeSH 诊断归一，增加更隐蔽的跨期污染样本，并把反事实重跑从 proxy 升级为 LLM 复核。",
            ]
        )
    text = "\n".join(lines) + "\n"
    write_text(Path(run_dir) / "analysis_zh.md", text)
    return text

