#!/bin/bash
# Build the TMT CID (ion trap) MS2 base model from PROSPECT PTMs: TMT (Zenodo record 8221499).
#
# Every number in README.md comes from the outputs of this script, written under $WORK:
#   prospect_stats.json    the dataset, and how it compares with the Prosit-TMT SI  (prospect_stats.py)
#   train_data/            training data + conversion.json          (prospect_to_carafe_training.py)
#   verify.log             conversion check                         (verify_prospect_conversion.py)
#   split/                 training, validation and held-out parts  (split_prosit_holdout.py)
#   environment.txt        hardware and software                    (../ms2hcd/environment.py)
#   heldout_reference.json the generic and the HCD TMT model on the held-out set
#   sweep/                 one model per tune_configs.tsv row, held-out scores, results.tsv
#   selected/              the selected run's loss curve and checksum, and runs G and H on one figure
#   prosit/                Prosit TMT (Koina, CID) vs the selected run, and mirror plots
#
# Usage: WORK=/path/to/workdir SI_XLSX=/path/to/ac1c05435_si_001.xlsx PY=/path/to/python ./run_ms2cid.sh
#   PROSPECT=dir     where the Zenodo files are (default: $WORK/prospect); downloaded and checked
#                    by MD5 if missing, checked by MD5 if present
#   SWEEP_ONLY=A,B   train only these rows of tune_configs.tsv (default: all)
#   SELECTED=D       the run to plot and compare with Prosit (default: D)
# SI_XLSX is Supplementary Table 1 of the Prosit-TMT paper (Anal. Chem. 2022, 94, 7181-7190,
# doi:10.1021/acs.analchem.1c05435), downloaded from the article page; its SHA-256 is checked.
# Needs ~37 GB for the Zenodo files plus ~5 GB in $WORK, a CUDA GPU, internet access (Zenodo, and
# Koina in step 11) and Python with peptdeep/alphabase/torch (the environment Carafe uses),
# matplotlib and openpyxl.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HCD="$HERE/../ms2hcd"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${WORK:?set WORK to a working directory}"
SI_XLSX="${SI_XLSX:?set SI_XLSX to the Prosit-TMT Supplementary Table 1 (ac1c05435_si_001.xlsx)}"
PY="${PY:-python}"
DEVICE="${DEVICE:-gpu}"
SELECTED="${SELECTED:-D}"
export PYTHONDONTWRITEBYTECODE=1
# The HCD TMT model (a release asset, not in git) is a starting point and a reference: check it first.
[ -s "$REPO/models/ms2tmt.pt" ] || { echo "missing $REPO/models/ms2tmt.pt (the HCD TMT model)"; exit 1; }
mkdir -p "$WORK"
PROSPECT="$(mkdir -p "${PROSPECT:-$WORK/prospect}" && cd "${PROSPECT:-$WORK/prospect}" && pwd)"
SI_XLSX="$(cd "$(dirname "$SI_XLSX")" && pwd)/$(basename "$SI_XLSX")"
cd "$WORK"
step() { echo; echo "== $* ($(date '+%H:%M:%S'))"; }

step "1. download and check PROSPECT PTMs: TMT (Zenodo 8221499)"
"$PY" "$HERE/download_prospect.py" --out-dir "$PROSPECT"

step "2. check the Prosit-TMT supplementary table"
expected=$(cut -d' ' -f1 "$HERE/prosit_tmt_si_checksum.txt")
[ "$(sha256sum "$SI_XLSX" | cut -d' ' -f1)" = "$expected" ] || { echo "$SI_XLSX: SHA-256 differs from prosit_tmt_si_checksum.txt"; exit 1; }
echo "SHA-256 matches prosit_tmt_si_checksum.txt"

step "3. describe the dataset"
[ -s prospect_stats.json ] || "$PY" "$HERE/prospect_stats.py" --prospect-dir "$PROSPECT" --si "$SI_XLSX" \
  --out prospect_stats.json > prospect_stats.txt

step "4. convert the CID spectra to training data"
[ -s train_data/conversion.json ] || "$PY" "$HERE/prospect_to_carafe_training.py" --prospect-dir "$PROSPECT" \
  --out-dir train_data --workers "$(nproc)" 2>&1 | tee convert.log

step "5. verify the conversion"
grep -q "VERDICT: conversion CORRECT" verify.log 2>/dev/null || \
  "$PY" "$HERE/verify_prospect_conversion.py" --prospect-dir "$PROSPECT" --train-dir train_data 2>&1 | tee verify.log
grep -q "VERDICT: conversion CORRECT" verify.log || { echo "conversion check failed; see verify.log"; exit 1; }

step "6. split: the supplementary table's hold_out sequences held out, 1% of the rest for validation (seed 2024)"
[ -s split/split.json ] || "$PY" "$HERE/split_prosit_holdout.py" --in-dir train_data --si "$SI_XLSX" \
  --out-dir split --valid-fraction 0.01 --seed 2024

step "7. record the environment"
"$PY" "$HCD/environment.py" --out environment.txt

step "8. reference: the generic AlphaPeptDeep model and the HCD TMT model on the held-out set"
[ -s heldout_reference.json ] || "$PY" "$HCD/evaluate_heldout.py" --heldout split/heldout --model generic \
  --model "hcd_tmt=$REPO/models/ms2tmt.pt" --device "$DEVICE" --out heldout_reference.json

step "9. train and score (tune_configs.tsv)"
"$PY" "$HCD/run_sweep.py" --split split --valid split/valid --configs "$HERE/tune_configs.tsv" --out-dir sweep \
  --start-model "hcd=$REPO/models/ms2tmt.pt" --python "$PY" --device "$DEVICE" ${SWEEP_ONLY:+--only "$SWEEP_ONLY"}

step "10. the selected run ($SELECTED): loss curve and checksum; runs G and H on one figure"
mkdir -p selected
"$PY" "$HCD/plot_loss_curve.py" --run "sweep/$SELECTED" --out selected/loss_curve \
  --title "CID model (run $SELECTED): training history" --lr-used-per-epoch --fine-ticks
sha256sum "sweep/$SELECTED/ms2_model.pt" | tee selected/ms2_model.sha256
# Runs G and H on one figure: the same settings from the HCD TMT model and from random weights.
if [ -s sweep/G/ms2_history.tsv ] && [ -s sweep/H/ms2_history.tsv ]; then
  "$PY" "$HERE/plot_loss_comparison.py" --run "G=sweep/G=the HCD TMT model" --run "H=sweep/H=random weights" \
    --out selected/loss_curve_G_vs_H \
    --title "CID model training history: run G (HCD TMT start) and run H (random start)"
fi

step "11. comparison with Prosit_2020_intensity_TMT (Koina, CID) on the held-out set, and mirror plots"
[ -s prosit/comparison.json ] && [ -s prosit/mirror_light.png ] || "$PY" "$HCD/compare_prosit_tmt.py" \
  --heldout split/heldout --train split/train --prospect-dir "$PROSPECT" --fragmentation CID \
  --model generic --model "$SELECTED=sweep/$SELECTED/ms2_model.pt" --device "$DEVICE" --out-dir prosit

step "done"
