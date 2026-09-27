#!/usr/bin/env python
"""Train the TMT MS2 base model from Carafe training data.

This wraps ``src/main/resources/py/v2/ai.py`` directly rather than going through
``carafe -tf ms2``. The Java entry point (AIGear.train_ms2_and_rt) forwards only a fixed set of
arguments and cannot pass ``--auto_tune``, ``--early_stop``, ``--use_best_model``,
``--ms2_epoch``, ``--ms2_lr`` or ``--ms2_batch_size``, all of which matter when training a base
model rather than fine-tuning on one experiment. Calling ai.py directly also keeps the Carafe
CLI untouched.

It then writes the model directory layout that Carafe reads back through ``-model_dir``:

    <model_dir>/ms2_model.pt
    <model_dir>/model_evaluation_metrics.json   (written by ai.py; the script reports whether
                                                 ms2.use_finetuned_for_prediction came out true)
    <model_dir>/meta.json                       (copied from the training data directory)

Note that Carafe overwrites the command line -nce from meta.json when a model directory is used
for library generation (AIGear.java, the -model_dir + -db branch), and the meta.json written by
the generic DDA path records one global NCE and an empty instrument. Edit the copied meta.json,
and pass -ms_instrument explicitly, before generating a library from this model.

Example
-------
    python train_tmt_ms2.py \
        --in-dir  tmt_train_data \
        --out-dir tmt_ms2_model \
        --instrument Lumos \
        --nce 30 \
        --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

# tools/tmt/ms2hcd/train_tmt_ms2.py -> repository root is three levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
AI_PY = os.path.join(REPO_ROOT, "src", "main", "resources", "py", "v2", "ai.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", required=True, help="Carafe output dir (psm_pdv.txt, fragment_intensity_df.tsv)")
    parser.add_argument("--out-dir", required=True, help="Where to write the trained model")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to run ai.py with")
    parser.add_argument("--ai-py", default=AI_PY, help=f"Path to ai.py (default: {AI_PY})")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--instrument",
        default="Lumos",
        help="AlphaPeptDeep instrument. The ProteomeTools TMT data is Orbitrap Fusion Lumos.",
    )
    parser.add_argument(
        "--nce",
        type=float,
        default=30.0,
        help="NCE used when psm_pdv.txt has no nce column at all. ai.py only fills the column "
        "when it is entirely absent, so a row with a blank nce cell stays NaN rather than "
        "picking this up -- make sure every row has an NCE before training.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--ms2-model",
        default=None,
        help="Optional starting checkpoint. Omit to start from the installed generic model.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random initialization. Not recommended: starting from the generic model converges faster.",
    )
    parser.add_argument(
        "--valid-dir",
        default=None,
        help="Optional validation set (psm_pdv.txt + fragment_intensity_df.tsv). Then ai.py trains on "
        "all of --in-dir and validates on this set, instead of splitting --in-dir itself.",
    )
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--force-use-finetuned",
        action="store_true",
        help="Force ms2.use_finetuned_for_prediction = true in model_evaluation_metrics.json. "
        "ai.py sets this itself only when the trained model beats the generic model on all of "
        "PCC/COS/SA/SPC, so only force it if you know why it did not.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args = parser.parse_args(argv)

    if not os.path.exists(args.ai_py):
        raise SystemExit(f"ai.py not found at {args.ai_py}")
    for directory in [args.in_dir] + ([args.valid_dir] if args.valid_dir else []):
        for required in ("psm_pdv.txt", "fragment_intensity_df.tsv"):
            path = os.path.join(directory, required)
            if not os.path.exists(path):
                raise SystemExit(f"missing training input {path}")
    # ai.py skips a starting checkpoint that does not exist and trains from the generic model
    # instead, without an error; a typo here would otherwise go unnoticed.
    if args.ms2_model and not os.path.isfile(args.ms2_model):
        raise SystemExit(f"starting checkpoint not found: {args.ms2_model}")

    os.makedirs(args.out_dir, exist_ok=True)

    cmd = [
        args.python,
        "-u",
        args.ai_py,
        "--in_dir", args.in_dir,
        "--out_dir", args.out_dir,
        "--out_prefix", "tmt_ms2",
        "--tf_type", "ms2",
        "--mode", "general",
        "--device", args.device,
        "--instrument", args.instrument,
        "--nce", str(args.nce),
        "--seed", str(args.seed),
        "--ms2_epoch", str(args.epochs),
        "--ms2_batch_size", str(args.batch_size),
        "--ms2_lr", str(args.lr),
        "--ms2_warmup_epoch", str(args.warmup_epochs),
        "--ms2_max_test_samples", str(args.max_test_samples),
        "--patience", str(args.patience),
        "--use_best_model",
    ]
    if not args.no_early_stop:
        cmd.append("--early_stop")
    if args.threads is not None:
        cmd += ["--threads", str(args.threads)]
    if args.ms2_model:
        cmd += ["--ms2_model", args.ms2_model]
    if args.valid_dir:
        cmd += ["--valid_dir", args.valid_dir]
    if args.from_scratch:
        cmd.append("--from_scratch")

    print("running:\n  " + " ".join(cmd), file=sys.stderr)
    if args.dry_run:
        return 0

    # ai.py imports models.py from src/main/resources/py/v2, and CPython would drop a
    # __pycache__ there. Maven's resource filtering then fails on the binary .pyc with
    # "MalformedInputException: Input length = 1", breaking the Java build.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ai.py exited with {result.returncode}", file=sys.stderr)
        # Exit with ai.py's own code, so run_sweep.py can recognise a native crash and retry. Python
        # takes a Windows code above 2**31 (e.g. 0xC0000409) only in its signed form.
        raise SystemExit(result.returncode - 2**32 if result.returncode >= 2**31 else result.returncode)

    # Carafe reads a model directory back through -model_dir, and ai_pred.py only loads the
    # trained checkpoint when model_evaluation_metrics.json says so. ai.py sets that flag itself
    # by comparing the trained model against the generic model on PCC/COS/SA/SPC, so report its
    # decision rather than quietly overriding it.
    metrics_path = os.path.join(args.out_dir, "model_evaluation_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

    ms2_metrics = metrics.get("ms2", {})
    chosen = bool(ms2_metrics.get("use_finetuned_for_prediction", False))
    selected = "trained" if chosen else "generic"
    print("", file=sys.stderr)
    print(f"ai.py selected the {selected} MS2 model for prediction", file=sys.stderr)
    for stage in ("pretrained_metrics", "finetuned_metrics"):
        if stage in ms2_metrics:
            values = ms2_metrics[stage]
            rendered = "  ".join(f"{k.upper()}={v:.4f}" for k, v in values.items() if isinstance(v, (int, float)))
            print(f"  {stage:<19} {rendered}", file=sys.stderr)

    if not chosen:
        if args.force_use_finetuned:
            metrics.setdefault("ms2", {})["use_finetuned_for_prediction"] = True
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            print("forced ms2.use_finetuned_for_prediction = true as requested", file=sys.stderr)
        else:
            print(
                "WARNING: the trained model did not beat the generic model on every metric, so "
                "Carafe will keep using the generic model. Investigate before shipping this "
                "model, or re-run with --force-use-finetuned if the reason is understood.",
                file=sys.stderr,
            )

    meta_src = os.path.join(args.in_dir, "meta.json")
    meta_dst = os.path.join(args.out_dir, "meta.json")
    if os.path.exists(meta_src) and not os.path.exists(meta_dst):
        shutil.copyfile(meta_src, meta_dst)
        print(f"copied {meta_src} -> {meta_dst}", file=sys.stderr)

    model_path = os.path.join(args.out_dir, "ms2_model.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: expected {model_path} to exist after training", file=sys.stderr)
    else:
        print(f"model ready: {model_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
