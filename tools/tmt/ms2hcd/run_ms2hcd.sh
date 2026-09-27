#!/bin/bash
# Build the TMT HCD MS2 base model from the NIST CPTAC3 TMT spectral library (PXL000010).
#
# Every number in README.md comes from the outputs of this script, written under $WORK:
#   nist_checksum.txt      SHA-256 of the downloaded archive (checked against ./nist_checksum.txt)
#   library_stats.json     the library as distributed                  (library_stats.py)
#   train_data/            training data + msp_conversion.json          (msp_to_carafe_training.py)
#   verify.log             conversion check                             (verify_msp_conversion.py)
#   split/                 training part + fixed held-out test set      (split_heldout.py)
#   environment.txt        hardware and software                        (environment.py)
#   sweep/                 one model per tune_configs.tsv row, held-out scores, results.tsv
#   heldout_reference.json the generic model on the held-out set
#   selected/              the selected run scored at fixed NCEs, and its loss curve
#   prosit/                Prosit TMT (Koina) vs the selected run, and mirror plots (compare_prosit_tmt.py)
#
# Usage: WORK=/path/to/workdir PY=/path/to/python ./run_ms2hcd.sh
#   SWEEP_ONLY=A,B   train only these rows of tune_configs.tsv (default: all)
#   SELECTED=D       the run to score at fixed NCEs and plot (default: D, the shipped model)
# Needs ~10 GB in $WORK, a CUDA GPU (DEVICE=cpu works but is far slower), and Python with
# peptdeep/alphabase/torch (the environment Carafe uses) and matplotlib. Step 10 sends the held-out
# peptides and 2,000 training peptides (public NIST library entries) to the public Koina server,
# so it needs internet access.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:?set WORK to a working directory}"
PY="${PY:-python}"
DEVICE="${DEVICE:-gpu}"
SELECTED="${SELECTED:-D}"
URL="https://chemdata.nist.gov/download/peptide_library/libraries/human/HCD/2022_03_16/cptac3_tmt_selected_passed_best.msp.tar.gz"
ARCHIVE="cptac3_tmt_selected_passed_best.msp.tar.gz"
MSP="cptac3_tmt_selected_passed_best.msp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK"
cd "$WORK"
step() { echo; echo "== $* ($(date '+%H:%M:%S'))"; }

step "1. download and check the archive"
expected=""
[ -s "$HERE/nist_checksum.txt" ] && expected=$(cut -d' ' -f1 "$HERE/nist_checksum.txt")
archive_ok() { [ -s "$ARCHIVE" ] && [ -n "$expected" ] && [ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$expected" ]; }
if ! archive_ok; then
  # Missing, partial, or not yet verified: fetch, resuming a partial file where it stopped.
  curl -L -C - --retry 10 -o "$ARCHIVE" "$URL" || echo "curl did not complete; checking the archive"
fi
sha256sum "$ARCHIVE" | tee nist_checksum.txt
if [ -n "$expected" ]; then
  [ "$(cut -d' ' -f1 nist_checksum.txt)" = "$expected" ] || {
    echo "checksum mismatch: expected $expected; delete $ARCHIVE and run again"; exit 1; }
  echo "checksum matches $HERE/nist_checksum.txt"
fi
# Extract under a temporary name, so an interrupted extraction is redone rather than used.
[ -s "$MSP" ] || { tar -xzOf "$ARCHIVE" "$MSP" > "$MSP.part" && mv "$MSP.part" "$MSP"; }

step "2. describe the library"
[ -s library_stats.json ] || "$PY" "$HERE/library_stats.py" --msp "$MSP" --out library_stats.json | tee library_stats.txt

step "3. convert to training data"
# No length or charge filter (the library spans length 6-50 and charge 1-8): keep every spectrum
# with at least one clean b/y ion, which scaling to a maximum of 1 needs.
[ -s train_data/msp_conversion.json ] || "$PY" "$HERE/msp_to_carafe_training.py" --msp "$MSP" --out-dir train_data \
  --min-charge 1 --max-charge 99 --min-length 1 --max-length 999 --min-fragments 1 2>&1 | tee convert.log

step "4. verify the conversion"
# Re-run unless a completed check passed; a log cut short by an interruption does not count.
grep -q "VERDICT: conversion CORRECT" verify.log 2>/dev/null || \
  "$PY" "$HERE/verify_msp_conversion.py" --msp "$MSP" --train-dir train_data 2>&1 | tee verify.log
grep -q "VERDICT: conversion CORRECT" verify.log || { echo "conversion check failed; see verify.log"; exit 1; }

step "5. held-out split (5% of sequences, seed 2024)"
[ -s split/split.json ] || "$PY" "$HERE/split_heldout.py" --in-dir train_data --out-dir split --fraction 0.05 --seed 2024

step "6. record the environment"
"$PY" "$HERE/environment.py" --out environment.txt

step "7. reference: the generic AlphaPeptDeep model on the held-out set"
[ -s heldout_reference.json ] || "$PY" "$HERE/evaluate_heldout.py" --heldout split/heldout --model generic \
  --device "$DEVICE" --out heldout_reference.json

step "8. train and score (tune_configs.tsv)"
"$PY" "$HERE/run_sweep.py" --split split --out-dir sweep --python "$PY" --device "$DEVICE" \
  ${SWEEP_ONLY:+--only "$SWEEP_ONLY"}

step "9. the selected run ($SELECTED): scores at fixed NCEs, and its loss curve"
mkdir -p selected
for nce in 27 30 32 38; do
  [ -s "selected/nce$nce.json" ] || "$PY" "$HERE/evaluate_heldout.py" --heldout split/heldout \
    --model "$SELECTED=sweep/$SELECTED/ms2_model.pt" --nce "$nce" --device "$DEVICE" --out "selected/nce$nce.json"
done
"$PY" "$HERE/plot_loss_curve.py" --run "sweep/$SELECTED" --out selected/loss_curve \
  --title "Shipped model (run $SELECTED): training history"
sha256sum "sweep/$SELECTED/ms2_model.pt" | tee selected/ms2_model.sha256

step "10. comparison with Prosit_2020_intensity_TMT (Koina) on the held-out set, and mirror plots"
[ -s prosit/comparison.json ] && [ -s prosit/mirror_light.png ] || "$PY" "$HERE/compare_prosit_tmt.py" \
  --heldout split/heldout --train split/train --msp "$MSP" --model generic \
  --model "$SELECTED=sweep/$SELECTED/ms2_model.pt" --device "$DEVICE" --out-dir prosit

step "done"
