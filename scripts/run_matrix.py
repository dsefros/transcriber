#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


PROMPTS = [
    "analysis/v4.yaml",
]

LOG_DIR = "logs"
LOG_FILE_NAME = "run_matrix.log"
SUMMARY_CSV_NAME = "run_matrix_summary.csv"


@dataclass
class RunResult:
    index: int
    total: int
    model: str
    prompt: str
    return_code: int
    status: str


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_log_dir(repo_root: Path) -> Path:
    log_dir = repo_root / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_log(log_file: Path, message: str, also_print: bool = True) -> None:
    line = f"[{timestamp()}] {message}"
    if also_print:
        print(line)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_raw(log_file: Path, message: str, also_print: bool = True) -> None:
    if also_print:
        print(message)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def stream_subprocess_output(proc: subprocess.Popen, log_file: Path) -> None:
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        write_raw(log_file, line, also_print=True)


def save_summary_csv(csv_path: Path, results: list[RunResult]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "total",
                "model",
                "prompt",
                "return_code",
                "status",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.index,
                    item.total,
                    item.model,
                    item.prompt,
                    item.return_code,
                    item.status,
                ]
            )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: PYTHONPATH=. python scripts/run_matrix.py <input_segments_json>")
        return 2

    input_file = Path(sys.argv[1]).resolve()
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    models_yaml_path = repo_root / "models.yaml"

    if not models_yaml_path.exists():
        print(f"models.yaml not found: {models_yaml_path}")
        return 2

    log_dir = ensure_log_dir(repo_root)
    log_file = log_dir / LOG_FILE_NAME
    summary_csv = log_dir / SUMMARY_CSV_NAME

    original_config = load_yaml(models_yaml_path)
    profiles = original_config.get("profiles")

    if not isinstance(profiles, dict) or not profiles:
        write_log(log_file, "ERROR: No profiles found in models.yaml")
        return 2

    model_keys = list(profiles.keys())
    total_runs = len(model_keys) * len(PROMPTS)
    run_index = 0
    results: list[RunResult] = []

    write_log(log_file, "=" * 80)
    write_log(log_file, "MATRIX RUN STARTED")
    write_log(log_file, f"Input segments file: {input_file}")
    write_log(log_file, f"Models found: {len(model_keys)}")
    write_log(log_file, f"Prompts found: {len(PROMPTS)}")
    write_log(log_file, f"Total runs: {total_runs}")
    write_log(log_file, "=" * 80)

    try:
        for model_key in model_keys:
            for prompt_path in PROMPTS:
                run_index += 1

                config = copy.deepcopy(original_config)
                config["default_model"] = model_key
                config["default_analysis_prompt"] = prompt_path
                save_yaml(models_yaml_path, config)

                write_log(log_file, "")
                write_log(log_file, "-" * 80)
                write_log(log_file, f"[{run_index}/{total_runs}]")
                write_log(log_file, f"model : {model_key}")
                write_log(log_file, f"prompt: {prompt_path}")
                write_log(
                    log_file,
                    f"cmd   : PYTHONPATH=. python -m src.app.cli {input_file}",
                )
                write_log(log_file, "-" * 80)

                env = os.environ.copy()
                env["PYTHONPATH"] = "."
                env.pop("ACTIVE_MODEL_PROFILE", None)

                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "src.app.cli",
                        str(input_file),
                    ],
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                try:
                    stream_subprocess_output(proc, log_file)
                finally:
                    returncode = proc.wait()

                status = "OK" if returncode == 0 else "FAILED"
                results.append(
                    RunResult(
                        index=run_index,
                        total=total_runs,
                        model=model_key,
                        prompt=prompt_path,
                        return_code=returncode,
                        status=status,
                    )
                )

                write_log(log_file, f"return_code: {returncode}")
                write_log(log_file, f"status     : {status}")

                if returncode != 0:
                    write_log(
                        log_file,
                        f"Run failed, but matrix will continue: "
                        f"model={model_key}, prompt={prompt_path}",
                    )

    finally:
        save_yaml(models_yaml_path, original_config)
        write_log(log_file, "Original models.yaml restored")

    ok_count = sum(1 for x in results if x.status == "OK")
    failed_count = sum(1 for x in results if x.status == "FAILED")

    write_log(log_file, "")
    write_log(log_file, "=" * 80)
    write_log(log_file, "MATRIX RUN SUMMARY")
    write_log(log_file, f"Total : {len(results)}")
    write_log(log_file, f"OK    : {ok_count}")
    write_log(log_file, f"FAILED: {failed_count}")
    write_log(log_file, "=" * 80)

    for item in results:
        write_log(
            log_file,
            f"[{item.index}/{item.total}] {item.model} | {item.prompt} | "
            f"{item.status} | return_code={item.return_code}",
        )

    save_summary_csv(summary_csv, results)
    write_log(log_file, f"Summary CSV saved: {summary_csv}")
    write_log(log_file, "MATRIX RUN FINISHED")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())