"""Single Colab command. Preserve failures and raw outputs; stop dependent work on failure."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.common import sha256, utc_now, write_json


def run_command(command, directory, name, allowed=(0,)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"started": utc_now(), "argv": command}
    raw = directory / f"{name}.raw.log"
    stamped = directory / f"{name}.timestamped.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               WANDB_DISABLED="true", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    with raw.open("wb") as raw_handle, stamped.open("w", encoding="utf-8") as stamp_handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for line in iter(process.stdout.readline, b""):
            raw_handle.write(line)
            raw_handle.flush()
            decoded = line.decode("utf-8", errors="replace")
            stamp_handle.write(f"{utc_now()} {decoded}")
            stamp_handle.flush()
            print(decoded, end="", flush=True)
        returncode = process.wait()
    record.update({"finished": utc_now(), "returncode": returncode, "accepted_exit_codes": list(allowed),
                   "raw_log_sha256": sha256(raw), "passed": returncode in allowed})
    write_json(directory / f"{name}.command.json", record)
    if returncode not in allowed:
        raise RuntimeError(f"{name} exited {returncode}; raw log retained at {raw}")
    return returncode


def gpu_sampler(directory, stop):
    query = ["nvidia-smi", "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    with (directory / "nvidia-smi.samples.raw.csv").open("wb") as output:
        while not stop.is_set():
            sample = subprocess.run(query, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            output.write(sample.stdout)
            output.flush()
            stop.wait(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="main", help="Prefer resolved SHA for a rerun")
    args = parser.parse_args()
    run_id = "t4-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = Path("evidence") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json("evidence/latest_run.json", {"path": str(directory), "status": "RUNNING"})
    status = {"run": run_id, "started": utc_now(), "status": "RUNNING", "completed": False}
    write_json(directory / "status.json", status)
    stop, sampler = threading.Event(), None
    py = [sys.executable, "-m"]
    try:
        run_command(["nvidia-smi"], directory, "nvidia-smi-before")
        run_command([sys.executable, "-m", "pip", "freeze"], directory, "pip-freeze")
        run_command(py + ["scripts.environment", "--output", str(directory / "environment.json")], directory, "environment")
        run_command(py + ["pytest", "-q", "-rs"], directory, "engineering-tests")
        run_command(py + ["ruff", "check", "scripts", "tests"], directory, "lint")
        sampler = threading.Thread(target=gpu_sampler, args=(directory, stop), daemon=True)
        sampler.start()
        run_command(py + ["scripts.build_data"], directory, "build-data")
        run_command(py + ["scripts.prepare", "--revision", args.revision, "--evidence", str(directory)], directory, "prepare")
        run_command(py + ["scripts.memory_budget", "--output", str(directory / "memory_budget.json")], directory, "memory-before")
        run_command(py + ["scripts.audit_data", "--output", str(directory / "data_audit.json")], directory, "audit-data")
        for name, command in [
            ("soup-doctor", ["soup", "doctor"]),
            ("soup-data-doctor", ["soup", "data", "doctor", "data/chat_for_doctor.jsonl", "--model", "models/base", "--max-length", "512", "--sample", "400", "--train-on-eot", "--show-mask", "2", "--output", str(directory / "soup_data_doctor.json")]),
            ("soup-data-lint", ["soup", "data", "lint", "data/train.jsonl", "--model", "models/base", "--sample", "400", "--output", str(directory / "soup_data_lint.json")]),
            ("soup-preflight", ["soup", "train", "--config", "config/soup.yaml", "--dry-run"]),
        ]:
            # data checks can report MAJOR (2); retain it and let report gate refuse shipping.
            run_command(command, directory, name, allowed=(0, 2) if "data-" in name else (0,))
        run_command(py + ["scripts.gradient_parity", "--output", str(directory / "gradient_parity.json")], directory, "gradient-parity")
        control = directory / "control"
        run_command(py + ["scripts.train", "--control", "--evidence", str(control)], directory, "control-train")
        run_command(py + ["scripts.verify_training", "--initial", str(control / "initial_adapter"), "--adapter", "outputs/control_adapter", "--limit", "4", "--output", str(control / "verification.json")], directory, "control-verify", allowed=(2,))
        control_result = json.loads((control / "verification.json").read_text(encoding="utf-8"))
        if control_result["tensor_delta"]["passed"] or control_result["functional_change"]["passed"]:
            raise RuntimeError("No-op control rejected for wrong reason; state/activation check failed to detect it")
        run_command(py + ["scripts.train", "--evidence", str(directory)], directory, "train")
        run_command(py + ["scripts.verify_training", "--initial", str(directory / "initial_adapter"), "--adapter", "outputs/adapter", "--output", str(directory / "verification.json")], directory, "verify-training")
        run_command(py + ["scripts.evaluate", "--verification", str(directory / "verification.json"), "--evidence", str(directory)], directory, "evaluate")
        run_command(["soup", "ship", "--evidence", str(directory / "ship_evidence.json"), "--output", str(directory / "soup_ship.json"), "--forgetting-threshold", "0.05"], directory, "soup-ship", allowed=(0, 2))
        status.update(status="COMPLETED", completed=True)
    except BaseException as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        print(f"Run failed; all earlier evidence remains: {exc}", file=sys.stderr)
        raise
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=5)
        status["finished"] = utc_now()
        write_json(directory / "status.json", status)
        write_json("evidence/latest_run.json", {"path": str(directory), "status": status["status"]})
        if shutil.which("nvidia-smi"):
            run_command(["nvidia-smi"], directory, "nvidia-smi-after")
        # Report even failed attempts, never silently replace the previous run.
        subprocess.run(py + ["scripts.make_report", "--evidence", str(directory)], check=False)


if __name__ == "__main__":
    main()
