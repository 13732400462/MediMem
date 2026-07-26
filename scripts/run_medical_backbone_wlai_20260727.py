from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


MODELS = (
    ("llama-3.1-70b", "llama_3_1_70b"),
    ("deepseek-v3", "deepseek_v3"),
    ("gpt-4.1", "gpt_4_1"),
    ("gemini-2.5-flash", "gemini_2_5_flash"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def api_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def probe_api(base_url: str, key: str, run_root: Path, timeout: int) -> dict[str, Any]:
    query_time = datetime.now(timezone.utc).isoformat()
    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers=api_headers(key),
        timeout=timeout,
    )
    raw = response.content
    models_audit: dict[str, Any] = {
        "queried_at": query_time,
        "http_status": response.status_code,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }
    response.raise_for_status()
    payload = response.json()
    model_ids = sorted(
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    models_audit.update(
        {
            "model_count": len(model_ids),
            "target_catalogue_presence": {
                model_id: model_id in model_ids for model_id, _ in MODELS
            },
        }
    )
    write_json(run_root / "models_query_audit.json", models_audit)

    probes = {}
    for model_id, _ in MODELS:
        started = time.time()
        completion = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=api_headers(key),
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "Return exactly OK."},
                    {"role": "user", "content": "ping"},
                ],
                "temperature": 0,
                "max_tokens": 8,
                "seed": 20260606,
            },
            timeout=timeout,
        )
        probe: dict[str, Any] = {
            "model_requested": model_id,
            "http_status": completion.status_code,
            "latency_s": round(time.time() - started, 3),
            "response_sha256": hashlib.sha256(completion.content).hexdigest(),
        }
        if completion.status_code == 200:
            body = completion.json()
            text = str(
                (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                or ""
            ).strip()
            usage = body.get("usage") or {}
            probe.update(
                {
                    "model_returned": body.get("model"),
                    "nonempty_output": bool(text),
                    "usage_present": bool(usage),
                    "usage": {
                        key_name: int(usage.get(key_name, 0) or 0)
                        for key_name in ("prompt_tokens", "completion_tokens", "total_tokens")
                    },
                }
            )
        else:
            probe["error_type"] = "http_error"
        probes[model_id] = probe
    write_json(run_root / "live_probe_audit.json", probes)
    failed = [
        model
        for model, probe in probes.items()
        if probe["http_status"] != 200
        or not probe.get("nonempty_output")
        or not probe.get("usage_present")
    ]
    if failed:
        raise RuntimeError(f"Target model live probes failed: {failed}")
    return {"models_query": models_audit, "probes": probes}


def monitor_gpu(run_root: Path, stop: threading.Event, gpu_index: int) -> None:
    path = run_root / "gpu_utilization.csv"
    path.write_text(
        "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,utilization_memory_pct\n",
        encoding="utf-8",
    )
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat()},{result.stdout.strip()}\n"
                )
        stop.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/table1_medical_5x500_20260720.jsonl"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(f"runs/medical_backbone_wlai_20260727_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/qwen3/bin/python"),
    )
    parser.add_argument("--base-url", default="https://api.wlai.vip/v1")
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("/root/.config/medimem/wlai.key"),
    )
    parser.add_argument("--workers-per-model", type=int, default=16)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    dataset = resolve(args.dataset)
    run_root = resolve(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "logs").mkdir()
    (run_root / "models").mkdir()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not args.key_file.is_file():
        raise FileNotFoundError(args.key_file)
    key = args.key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("WLAI key file is empty.")
    probe_api(args.base_url, key, run_root, timeout=60)

    frozen = {
        "protocol": "Section 4.2 medical backbone comparison, frozen 5x500",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(args.project),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "case_count": 2500,
        "models": [model for model, _ in MODELS],
        "pipelines": ["Direct", "CoT", "A-MEM", "CliCARE", "MediMem"],
        "baseline_set": "backbone",
        "ablation_groups": ["full"],
        "run_seed": 20260606,
        "counterfactual_policy": "risk_sample",
        "counterfactual_sample_rate": 0.2,
        "counterfactual_risk_threshold": 0.55,
        "medical_prediction_max_tokens": 700,
        "workers_per_model": args.workers_per_model,
        "base_url": args.base_url,
        "api_key_file": str(args.key_file),
        "git_commit": args.git_commit,
        "strict_no_fallback": True,
    }
    write_json(run_root / "frozen_config.json", frozen)
    (run_root / "FORMAL_STARTED").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )

    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(run_root, stop_monitor, args.gpu_index),
        daemon=True,
    )
    monitor.start()
    processes = []
    for model_id, slug in MODELS:
        model_root = run_root / "models" / slug
        model_root.mkdir()
        log_path = run_root / "logs" / f"{slug}.log"
        command = [
            str(args.python),
            "-u",
            "-m",
            "mem_ehr_agent",
            "experiment-suite",
            "--dataset",
            str(dataset),
            "--require-api",
            "--max-workers",
            str(args.workers_per_model),
            "--suite-profile",
            "fast-formal",
            "--baseline-set",
            "backbone",
            "--ablation-groups",
            "full",
            "--counterfactual-policy",
            "risk_sample",
            "--counterfactual-sample-rate",
            "0.20",
            "--counterfactual-risk-threshold",
            "0.55",
            "--defer-reports",
            "--run-seed",
            "20260606",
            "--output-root",
            str(model_root),
        ]
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(args.project),
                "DEEPSEEK_BASE_URL": args.base_url,
                "DEEPSEEK_API_KEY": "",
                "DEEPSEEK_API_KEY_FILE": str(args.key_file),
                "DEEPSEEK_MODEL": model_id,
                "DEEPSEEK_TIMEOUT": "600",
                "DEEPSEEK_MAX_TOKENS": "1800",
                "MEDICAL_PREDICTION_MAX_TOKENS": "700",
                "MEDICAL_JSON_PARSE_RETRIES": "2",
                "MEDICAL_LLM_SEED": "20260606",
                "MEDICAL_STRICT_NO_LEAK_FILTER": "1",
                "FAST_FORMAL_EARLY_STOP_ON_WIN": "0",
                "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.project,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record = {
            "model": model_id,
            "slug": slug,
            "pid": process.pid,
            "command": command,
            "log": str(log_path),
            "model_root": str(model_root),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(model_root / "launch.json", record)
        processes.append((record, process, log_handle))

    results = []
    for record, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        run_dirs = sorted(
            (path for path in Path(record["model_root"]).iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
        result = {
            **record,
            "returncode": returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dirs[-1]) if run_dirs else None,
        }
        results.append(result)
        write_json(Path(record["model_root"]) / "result.json", result)

    stop_monitor.set()
    monitor.join(timeout=10)
    supervisor = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "runs": results,
        "passed": all(row["returncode"] == 0 and row["run_dir"] for row in results),
    }
    write_json(run_root / "supervisor_result.json", supervisor)
    if not supervisor["passed"]:
        raise RuntimeError("One or more medical backbone runs failed.")
    (run_root / "FORMAL_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(run_root)


if __name__ == "__main__":
    main()
