#!/usr/bin/env python3
"""Compare two Carafe spectral libraries (DIA-NN TSV) fragment-by-fragment.

Use this to answer "how similar is the local AlphaPepDeep prediction to the remote
(Koina) one?" for a given peptide. It aligns fragments by (FragmentType,
FragmentNumber, FragmentCharge) within a precursor and reports the cosine
similarity of the relative intensities.

Typical workflow (run on a machine with the Carafe Python venv installed):

  # 1) Make a one-peptide FASTA (NoCut => the entry is treated as a single peptide).
  printf '>sp|TEST|ELVISLIVESR\nELVISLIVESR\n' > elvis.fasta

  # 2) Local AlphaPepDeep prediction (uses ~/.carafe/.venv):
  java -jar carafe.jar -db elvis.fasta -o local_out -enzyme NoCut \
      -lf_type DIA-NN -nce 25 -ms_instrument Lumos \
      -min_pep_charge 2 -max_pep_charge 2 -python ~/.carafe/.venv/bin/python

  # 3) Remote Koina AlphaPepDeep prediction:
  java -jar carafe.jar -build_koina_library koina_lib.tsv -db elvis.fasta \
      -koina_ms2_model AlphaPeptDeep_ms2_generic -koina_rt_model AlphaPeptDeep_rt_generic \
      -nce 25 -ms_instrument Lumos -min_pep_charge 2 -max_pep_charge 2

  # 4) Compare:
  python compare_library_predictions.py \
      --local local_out/carafe_spectral_library.tsv \
      --koina koina_lib.tsv \
      --peptide ELVISLIVESR --charge 2
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load_fragments(path: Path, peptide: str, charge: int) -> dict[tuple, float]:
    """Map (FragmentType, FragmentNumber, FragmentCharge) -> RelativeIntensity for one precursor."""
    frags: dict[tuple, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("StrippedPeptide") != peptide:
                continue
            if int(float(row.get("PrecursorCharge", 0))) != charge:
                continue
            key = (row["FragmentType"], int(row["FragmentNumber"]), int(row["FragmentCharge"]))
            frags[key] = float(row["RelativeIntensity"])
    return frags


def cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two Carafe DIA-NN libraries for one peptide.")
    ap.add_argument("--local", required=True, type=Path, help="Local prediction library TSV.")
    ap.add_argument("--koina", required=True, type=Path, help="Koina prediction library TSV.")
    ap.add_argument("--peptide", required=True, help="Stripped peptide sequence, e.g. ELVISLIVESR.")
    ap.add_argument("--charge", type=int, default=2, help="Precursor charge to compare [default: 2].")
    ap.add_argument("--top", type=int, default=10, help="Show this many top fragments [default: 10].")
    args = ap.parse_args()

    local = load_fragments(args.local, args.peptide, args.charge)
    koina = load_fragments(args.koina, args.peptide, args.charge)
    if not local:
        print(f"WARNING: no fragments for {args.peptide} (z={args.charge}) in {args.local}")
    if not koina:
        print(f"WARNING: no fragments for {args.peptide} (z={args.charge}) in {args.koina}")

    cos = cosine(local, koina)
    print(f"\nPeptide {args.peptide} (z={args.charge})")
    print(f"  local fragments: {len(local)}   koina fragments: {len(koina)}")
    print(f"  cosine similarity (relative intensities): {cos:.4f}\n")

    keys = sorted(set(local) | set(koina), key=lambda k: -max(local.get(k, 0), koina.get(k, 0)))
    print(f"  {'fragment':<12} {'local':>8} {'koina':>8}")
    for k in keys[: args.top]:
        frag = f"{k[0]}{k[1]}+{k[2]}"
        print(f"  {frag:<12} {local.get(k, 0):>8.4f} {koina.get(k, 0):>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
