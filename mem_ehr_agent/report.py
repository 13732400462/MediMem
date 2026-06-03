from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import BASELINE_SOURCES
from .io_utils import write_text


def metric_cell(value: Any, *, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def render_report(
    *,
    run_dir: str | Path,
    data_notes: str,
    summaries: list[dict[str, Any]],
    optimization_log: list[dict[str, Any]],
    best_round: dict[str, Any] | None,
    blocker: str | None = None,
    feature_flags: dict[str, Any] | None = None,
    error_analysis: dict[str, Any] | None = None,
    leakage_audit: dict[str, int] | None = None,
    memory_op_confusion: dict[str, int] | None = None,
    pollution_type_breakdown: dict[str, dict[str, float]] | None = None,
    expected_op_distribution: dict[str, dict[str, int]] | None = None,
    slice_breakdown: list[dict[str, Any]] | None = None,
) -> str:
    notes = data_notes.strip() or "公开数据 API 正常，或当前运行使用了已有数据集文件。"
    present_methods = {str(row.get("method", "")) for row in summaries}
    ddo_included = any("ddo" in method for method in present_methods)
    colacare_included = any("colacare" in method for method in present_methods)
    lines = [
        "# 纵向 EHR 记忆增强多智能体实验报告",
        "",
        "## 数据来源与构建流程",
        "- 主数据源：PMOA-TTS（PubMed OA 病例时间序列）。",
        "- 辅数据源：PMC-Patients（PubMed Central 病例摘要/相似病例上下文）。",
        "- 本版本使用无泄露污染构造：运行时污染记忆不包含 gold label、expected op 或 revised claim。",
        "- `expected_memory_ops` 仅用于离线评价，不进入 ours 或 baseline 的推理链路。",
        "- 数据抓取备注：",
        "```text",
        notes,
        "```",
        "",
        "## Baseline 设置与口径",
        "- 当前医疗源实验主表采用 focused baseline 集合：A-MEM、Polluted A-MEM、Direct、Polluted Direct。",
        "- 因此结论只能表述为相对当前 focused baseline 集合的阶段性比较，不表述为完整横向 baseline 或先进性结论。",
        f"- A-MEM：{BASELINE_SOURCES['amem']['paper']}；官方 repo：{BASELINE_SOURCES['amem']['repo']}。",
        (
            f"- DDO：{BASELINE_SOURCES['ddo']['paper']}；官方 repo：{BASELINE_SOURCES['ddo']['repo']}。"
            if ddo_included
            else f"- DDO：{BASELINE_SOURCES['ddo']['paper']}；not included in this medical-source table，需同数据、同样本、同模型、同指标补跑后才能进入主表。"
        ),
        (
            f"- ColaCare：{BASELINE_SOURCES['colacare']['paper']}；官方 repo：{BASELINE_SOURCES['colacare']['repo']}。"
            if colacare_included
            else f"- ColaCare：{BASELINE_SOURCES['colacare']['paper']}；not included in this medical-source table，需同数据、同样本、同模型、同指标补跑后才能进入主表。"
        ),
        "- MemoryOS、MemInsight、GMemory 与官方 DDO wrapper：not included in this medical-source table；LoCoMo wrapper 横向结果需单独成表，不能混入医疗源诊断任务主表。",
        "- Direct DeepSeek：截断上下文、无长期记忆。",
        "- 原始 baseline：不接触污染记忆，用于诊断能力参考。",
        "- Polluted baseline：接触同样污染记忆，但没有 proposed cleaning module，用于污染鲁棒性主比较。",
        "- Ours：JSONL 动态记忆、LLM Critic 记忆清洗、动态 top_k 与反事实 proxy。",
        "",
        "## 当前渲染轮特性状态",
        "- 本节描述当前渲染轮的 feature flags；指标汇总可能同时包含 Full Ours 与多个 ablation merged rows。",
    ]
    flags = feature_flags or {}
    feature_rows = [
        ("诊断归一化", not bool(flags.get("disable_normalization"))),
        ("动态 top_k", not bool(flags.get("disable_dynamic_top_k"))),
        ("记忆清洗", not bool(flags.get("disable_memory_cleaning"))),
        ("Critic 操作护栏", not bool(flags.get("disable_critic_op_guard"))),
        ("Source evidence notes 注入", not bool(flags.get("disable_evidence_note_injection"))),
        ("Counterfactual verification", not bool(flags.get("disable_counterfactual_verification"))),
    ]
    for name, enabled in feature_rows:
        lines.append(f"- {name}：{'启用' if enabled else '关闭'}")
    lines.extend(
        [
            "",
            "## 指标汇总",
            "| Method | N | Primary Acc | Diagnosis F1 | CDR F1 | MPCS | CPG | CF Pass | Avg Tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {method} | {n} | {acc} | {f1} | {cdr} | {mpcs} | {cpg} | {cfpass} | {tok} |".format(
                method=row["method"],
                n=row["n"],
                acc=metric_cell(row.get("primary_diagnosis_top1_accuracy")),
                f1=metric_cell(row.get("diagnosis_list_f1")),
                cdr=metric_cell(row.get("cdr_f1")),
                mpcs=metric_cell(row.get("memory_pollution_control_score")),
                cpg=metric_cell(row.get("counterfactual_probability_gap")),
                cfpass=metric_cell(row.get("counterfactual_pass_rate")),
                tok=metric_cell(row.get("avg_tokens"), digits=1),
            )
        )
    lines.extend(
        [
            "",
            "说明：Hard Hit 是兼容旧版 Rule Suppression 的硬规则命中审计，只统计 Discard/Invalidate；"
            "主污染指标应看 ActionQ、RevisionQ、FactSoft、OverDel 和 MPCS。"
            "Baseline 若没有输出 memory_ops，操作级指标显示为 N/A，避免把不可评价误读为 0 分。",
            "消融解释必须采用多指标口径：去除某模块后 Primary Acc/CDR F1 可能升高，"
            "但若 MPCS、Revision 或 FactSoft 归零/显著下降，应解释为污染控制失败或安全性退化，"
            "不能据此判断该模块无意义。",
            "",
            "## 自动优化记录",
        ]
    )
    for item in optimization_log[-10:]:
        lines.append(
            f"- Round {item.get('round')}: strategy={item.get('strategy')} "
            f"ours_acc={float(item.get('ours_accuracy', 0)):.3f}, "
            f"best_baseline={float(item.get('best_baseline_accuracy', 0)):.3f}, won={item.get('won')}"
        )
    if best_round:
        lines.extend(
            [
                "",
                "## 最好轮次",
                f"- Round：{best_round.get('round')}",
                f"- Strategy：`{best_round.get('strategy')}`",
                f"- Ours primary accuracy：{float(best_round.get('ours_accuracy', 0)):.3f}",
                f"- Best baseline primary accuracy：{float(best_round.get('best_baseline_accuracy', 0)):.3f}",
            ]
        )
    if blocker:
        lines.extend(["", "## Blocker", f"- {blocker}"])
    if error_analysis:
        buckets = error_analysis.get("buckets", {})
        lines.extend(
            [
                "",
                "## 错误样本摘要",
                f"- Ours 错 / A-MEM 对：{len(buckets.get('ours_wrong_amem_right', []))}",
                f"- Ours 对 / A-MEM 错：{len(buckets.get('ours_right_amem_wrong', []))}",
                f"- 双方都错：{len(buckets.get('both_wrong', []))}",
                f"- 诊断列表 F1 偏低：{len(buckets.get('low_diagnosis_f1', []))}",
                f"- CDR F1 偏低：{len(buckets.get('low_cdr_f1', []))}",
                "- 详细列表见：`error_analysis_zh.md`。",
            ]
        )
    if leakage_audit:
        lines.extend(["", "## 泄露审计"])
        for key in [
            "runtime_gold_mentions",
            "prompt_memory_ops_gold_mentions",
            "counterfactual_runtime_gold_mentions",
            "poison_expected_op",
            "poison_revised_claim",
            "poison_expected_memory_ops",
            "source_real_false",
        ]:
            lines.append(f"- `{key}`：{int(leakage_audit.get(key, 0))}")
    if memory_op_confusion:
        lines.extend(["", "## 记忆操作混淆矩阵"])
        for key, count in memory_op_confusion.items():
            lines.append(f"- `{key}`：{count}")
    if expected_op_distribution:
        lines.extend(["", "## 预期操作分布"])
        lines.append("| Pollution Type | Expected Ops |")
        lines.append("|---|---|")
        for pollution_type, counts in expected_op_distribution.items():
            ops = ", ".join(f"{op}={count}" for op, count in counts.items())
            lines.append(f"| {pollution_type} | {ops} |")
    if pollution_type_breakdown:
        lines.extend(["", "## 污染类型分解"])
        lines.append("| Pollution Type | N | ActionQ | RevisionQ | FactSoft | OverDel | MPCS |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pollution_type, row in pollution_type_breakdown.items():
            lines.append(
                "| {ptype} | {n} | {action} | {rev} | {fact} | {over} | {mpcs} |".format(
                    ptype=pollution_type,
                    n=metric_cell(row.get("n"), digits=0),
                    action=metric_cell(row.get("action_quality")),
                    rev=metric_cell(row.get("revision_quality")),
                    fact=metric_cell(row.get("fact_preservation_soft")),
                    over=metric_cell(row.get("over_deletion_rate")),
                    mpcs=metric_cell(row.get("memory_pollution_control_score")),
                )
            )
    if slice_breakdown:
        lines.extend(["", "## 稳健性切片指标"])
        lines.append("| Slice | Method | N | Primary Acc | Diagnosis F1 | CDR F1 | MPCS | Avg Tokens |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in slice_breakdown:
            lines.append(
                "| {slice} | {method} | {n} | {acc} | {f1} | {cdr} | {mpcs} | {tok} |".format(
                    slice=row.get("slice"),
                    method=row.get("method"),
                    n=row.get("n"),
                    acc=metric_cell(row.get("primary_diagnosis_top1_accuracy")),
                    f1=metric_cell(row.get("diagnosis_list_f1")),
                    cdr=metric_cell(row.get("cdr_f1")),
                    mpcs=metric_cell(row.get("memory_pollution_control_score")),
                    tok=metric_cell(row.get("avg_tokens"), digits=1),
                )
            )
    exceeded = bool(best_round and best_round.get("won"))
    lines.extend(
        [
            "",
            "## 结论与下一步",
            f"- 相对当前 focused baseline 集合的最强 polluted baseline（按 Primary Acc）是否更高：{'是' if exceeded else '否'}。",
            "- 该判断仅覆盖当前医疗源 focused baseline 集合；DDO、ColaCare、MemoryOS、MemInsight、GMemory 等完整横向 baseline 需补跑后单独汇报。",
            "- 只有新无泄露 run 可作为主结论；旧 `stale_memory_50` / `partial_truth_pollution_50` 结果仅作为问题定位记录。",
            "- 若仅 Primary Acc 略高但 Diagnosis F1/CDR F1 或污染安全指标较低，应表述为部分指标更好，不能表述为整体领先。",
            "- 下一步建议报告 Stale Action、Revise Acc、Fact Keep 和 Over Delete，避免单独把污染抑制率解释为绝对安全。",
        ]
    )
    text = "\n".join(lines) + "\n"
    write_text(Path(run_dir) / "analysis_zh.md", text)
    return text
