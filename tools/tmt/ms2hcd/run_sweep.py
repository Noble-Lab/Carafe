#!/usr/bin/env python
"""Train one MS2 model per row of tune_configs.tsv and score each on the held-out test set.

Every run starts from the generic AlphaPeptDeep model, unless the configs' optional 'start' column
names another start ('scratch' for random initialization, or a checkpoint given with
--start-model), and trains on the training part written by split_heldout.py. Early stopping is
off, so each run completes its learning-rate schedule (warmup, then decay to 0) and schedules are
compared like for like; the saved weights are those of the best epoch on ai.py's internal
validation split, or on the --valid set when given, never chosen on the held-out set.

    python run_sweep.py --split split --out-dir sweep [--only A,C]

Writes sweep/<name>/ (model directory and train.log), sweep/<name>/heldout.json, and
sweep/results.tsv with one row per run: settings, wall-clock training time, epochs run and the
held-out medians. Runs whose heldout.json already exists are skipped, so the sweep can resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# On Windows, a CUDA process started seconds after another exits occasionally dies at start-up with
# a native fast-fail (0xC0000409) and no Python traceback. Each step waits before starting and is
# retried once after such a crash; a step starts from scratch, so a retry redoes the same work.
NATIVE_CRASH = 3221226505
GPU_PAUSE_SECONDS = 15


def run(cmd, log_path):
    for attempt in (1, 2):
        time.sleep(GPU_PAUSE_SECONDS)
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != NATIVE_CRASH or attempt == 2:
            return proc.returncode
        os.replace(log_path, log_path + ".attempt1")  # keep the crashed attempt's log
        print(f"native crash (exit {NATIVE_CRASH}) in {os.path.basename(cmd[1])}; retrying once "
              f"(first attempt's log: {log_path}.attempt1)", flush=True)


def training_dir(split_dir, scope):
    """Training data for a run: the whole training part ("all"), or only its spectra inside Carafe's
    default library range ("default", built once from the training part)."""
    if scope == "all":
        return os.path.join(split_dir, "train")
    if scope != "default":
        sys.exit(f"unknown scope {scope!r} in the configs (use 'all' or 'default')")
    out = os.path.join(split_dir, "train_default")
    if not os.path.isdir(out):
        import shutil
        sys.path.insert(0, HERE)
        from split_heldout import restrict_to_default_range
        # Built under a temporary name and renamed when complete, so an interrupted build is redone.
        tmp = out + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        kept, total = restrict_to_default_range(os.path.join(split_dir, "train"), tmp)
        os.replace(tmp, out)
        print(f"default-range training part: {kept} of {total} training spectra", flush=True)
    return out


def epochs_run(model_dir):
    path = os.path.join(model_dir, "ms2_history.tsv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return max(0, sum(1 for _ in fh) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="directory written by split_heldout.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", default=os.path.join(HERE, "tune_configs.tsv"))
    ap.add_argument("--only", help="comma-separated run names")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--valid", help="optional validation set passed to training (train_tmt_ms2.py --valid-dir)")
    ap.add_argument("--start-model", action="append", default=[], metavar="NAME=PATH",
                    help="a starting checkpoint that the configs' optional 'start' column can name "
                         "('generic', the default, is the installed AlphaPeptDeep model; 'scratch' is "
                         "random initialization)")
    args = ap.parse_args(argv)
    start_models = dict(spec.split("=", 1) for spec in args.start_model)

    with open(args.configs, newline="") as fh:
        all_configs = list(csv.DictReader(fh, delimiter="\t"))
    configs = all_configs
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in all_configs if c["name"] in wanted]
    os.makedirs(args.out_dir, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    os.environ.update(env)

    # 1. train and score the requested runs that have no results yet
    for c in configs:
        model_dir = os.path.join(args.out_dir, c["name"])
        heldout_json = os.path.join(model_dir, "heldout.json")
        if not os.path.exists(heldout_json):
            os.makedirs(model_dir, exist_ok=True)
            train = [args.python, os.path.join(HERE, "train_tmt_ms2.py"),
                     "--in-dir", training_dir(args.split, c.get("scope", "all")), "--out-dir", model_dir,
                     "--epochs", c["epochs"], "--lr", c["lr"], "--batch-size", c["batch_size"],
                     "--warmup-epochs", c["warmup_epochs"], "--max-test-samples", "5000",
                     "--no-early-stop", "--device", args.device, "--python", args.python]
            start_from = c.get("start") or "generic"
            if start_from == "scratch":
                train += ["--from-scratch"]
            elif start_from != "generic":
                if start_from not in start_models:
                    sys.exit(f"[{c['name']}] start model {start_from!r} is not defined (use --start-model)")
                train += ["--ms2-model", start_models[start_from]]
            if args.valid:
                train += ["--valid-dir", args.valid]
            print(f"[{c['name']}] training: epochs={c['epochs']} lr={c['lr']} batch={c['batch_size']}"
                  + (f" from {start_from}" if "start" in c else ""), flush=True)
            start = time.time()
            code = run(train, os.path.join(model_dir, "train.log"))
            seconds = time.time() - start
            if code != 0:
                sys.exit(f"[{c['name']}] training failed (exit {code}); see {model_dir}/train.log")
            with open(os.path.join(model_dir, "timing.json"), "w") as fh:
                json.dump({"training_wall_seconds": round(seconds, 1)}, fh)
            evaluate = [args.python, os.path.join(HERE, "evaluate_heldout.py"),
                        "--heldout", os.path.join(args.split, "heldout"),
                        "--model", f"{c['name']}={os.path.join(model_dir, 'ms2_model.pt')}",
                        "--device", args.device, "--out", heldout_json]
            code = run(evaluate, os.path.join(model_dir, "heldout.log"))
            if code != 0:
                sys.exit(f"[{c['name']}] evaluation failed (exit {code}); see {model_dir}/heldout.log")

    # 2. results.tsv from every run on disk, not only this invocation's (so --only never drops rows),
    #    read back from heldout.json so a re-scored run is reported with its current scores
    rows = []
    for c in all_configs:
        model_dir = os.path.join(args.out_dir, c["name"])
        heldout_json = os.path.join(model_dir, "heldout.json")
        if not os.path.exists(heldout_json):
            continue
        with open(heldout_json) as fh:
            result = json.load(fh)[0]
        with open(os.path.join(model_dir, "timing.json")) as fh:
            seconds = json.load(fh)["training_wall_seconds"]
        default = result.get("default_range", {})
        row = {**{k: c[k] for k in ("name", "epochs", "lr", "batch_size", "warmup_epochs")},
               "scope": c.get("scope", "all"),
               **({"start": c.get("start") or "generic"} if "start" in c else {}),
               "epochs_run": epochs_run(model_dir), "train_minutes": round(seconds / 60, 1),
               **{k: round(result[k], 4) for k in ("PCC", "COS", "SA", "SPC", "SA_gt_0.9")},
               # held-out spectra inside Carafe's default library range (length 7-35, charge 2-4)
               **{f"{k}_default": round(default[k], 4) for k in ("SA", "SA_gt_0.9") if k in default}}
        rows.append(row)
        print(f"[{c['name']}] held-out SA {row['SA']:.4f}  SA>0.9 {row['SA_gt_0.9']:.1%}  "
              f"({row['train_minutes']} min)", flush=True)

    if rows:
        with open(os.path.join(args.out_dir, "results.tsv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {os.path.join(args.out_dir, 'results.tsv')}")


if __name__ == "__main__":
    main()
