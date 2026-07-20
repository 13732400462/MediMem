from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--served-model", default="qwen3-vl-8b")
    args = parser.parse_args()

    files = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in args.model_path.rglob("*") if item.is_file()):
        relative = str(path.relative_to(args.model_path)).replace("\\", "/")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": value}
        )
        aggregate.update(relative.encode())
        aggregate.update(value.encode())
    report = {
        "model_path": str(args.model_path),
        "served_model": args.served_model,
        "artifact_sha256": aggregate.hexdigest(),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report["artifact_sha256"])


if __name__ == "__main__":
    main()
