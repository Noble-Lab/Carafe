#!/usr/bin/env python
"""Prove that an MSP conversion put every intensity in the right fragment cell.

``msp_to_carafe_training.py`` has to get four things right at once: the peptide sequence, the
modification positions (AlphaBase's ``mod_sites`` are 1-based with 0 for the peptide N-term,
while NIST uses 0-based residue indices), the fragment matrix row (``b_n`` at row n-1, ``y_n``
at row L-n-1) and the charge column. Any one of those being wrong still produces a
plausible-looking file, so this checks it from the outside.

Method
------
For every populated cell, ask AlphaBase for that cell's theoretical m/z given the sequence and
modifications we wrote, then find the observed peak in the MSP that we took the intensity from
and compare. Two things are reported:

1. the deviation between the observed peak m/z and AlphaBase's theoretical m/z, and
2. that deviation against the ppm error NIST itself states in the annotation
   (``"b5/0.6ppm"``).

The second is the strong test. NIST's stated ppm is computed from *its* theoretical m/z, so if
our masses agree with theirs the two numbers must agree too -- a wrong modification position
would shift a TMT-bearing fragment by 229 Da and show up immediately.

Records are matched by (sequence, charge, NCE) rather than by position, because the converter
skips spectra and MSP record *i* is therefore not output row *i*.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msp_to_carafe_training import parse_peak_annotation  # noqa: E402

CHARGE_RE = re.compile(r"\bCharge=(\d+)")
NCE_RE = re.compile(r"@hcd([\d.]+)")
PPM_RE = re.compile(r"/(-?[\d.]+)ppm")
FRAG_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msp", required=True, help="The source .msp library")
    parser.add_argument("--train-dir", required=True, help="Directory written by msp_to_carafe_training.py")
    parser.add_argument("--max-records", type=int, default=2000,
                        help="How many MSP records to scan (default: 2000)")
    parser.add_argument("--max-ppm", type=float, default=25.0,
                        help="Tolerance for an observed peak to count as matching (default: 25)")
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    import alphabase.peptide.fragment as fragment

    psm = pd.read_csv(os.path.join(args.train_dir, "psm_pdv.txt"), sep="\t",
                      dtype={"mods": str, "mod_sites": str})
    psm["mods"] = psm["mods"].fillna("")
    psm["mod_sites"] = psm["mod_sites"].fillna("")
    psm["nAA"] = psm["sequence"].str.len()
    frag = pd.read_csv(os.path.join(args.train_dir, "fragment_intensity_df.tsv"), sep="\t")

    if len(frag) != int(psm["frag_stop_idx"].max()):
        print(f"FAIL: fragment file has {len(frag)} rows but the last frag_stop_idx is "
              f"{int(psm['frag_stop_idx'].max())}", file=sys.stderr)
        return 1

    # Contiguity: blocks must tile the fragment table exactly.
    starts = psm["frag_start_idx"].to_numpy()
    stops = psm["frag_stop_idx"].to_numpy()
    lengths = psm["nAA"].to_numpy() - 1
    if not (stops - starts == lengths).all():
        print("FAIL: some blocks are not (peptide length - 1) rows", file=sys.stderr)
        return 1
    if starts[0] != 0 or not (starts[1:] == stops[:-1]).all():
        print("FAIL: fragment blocks are not contiguous", file=sys.stderr)
        return 1
    print(f"block layout OK: {len(psm)} PSMs tile {len(frag)} fragment rows contiguously")

    scan = psm.head(max(1, args.max_records)).copy()
    mz_df = fragment.create_fragment_mz_dataframe(
        scan[["sequence", "mods", "mod_sites", "charge", "nAA"]].copy(), FRAG_COLUMNS)

    key2row = {}
    for i, row in scan.iterrows():
        key2row.setdefault((row.sequence, int(row.charge), f"{float(row.nce):g}"), i)

    observed = {}
    name = None
    current = None
    seen = 0
    with io.open(args.msp, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Name:"):
                name = line[6:].strip().split("/")[0]
                current = None
                seen += 1
                if seen > args.max_records * 3:
                    break
            elif line.startswith("Comment:"):
                charge = CHARGE_RE.search(line)
                nce = NCE_RE.search(line)
                if charge and nce:
                    key = (name, int(charge.group(1)), f"{float(nce.group(1)):g}")
                    if key in key2row and key not in observed:
                        current = {}
                        observed[key] = current
            elif current is not None and line and line[0].isdigit():
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    parsed = parse_peak_annotation(parts[2], True)
                    if parsed and parsed[2] <= 2:
                        stated = PPM_RE.search(parts[2])
                        current.setdefault(parsed, []).append(
                            (float(parts[0]), float(stated.group(1)) if stated else None))

    ours, theirs, unmatched, checked = [], [], 0, 0
    for key, peaks in observed.items():
        i = key2row[key]
        row = scan.loc[i]
        start, stop = int(row.frag_start_idx), int(row.frag_stop_idx)
        n_rows = int(row.nAA) - 1
        block = frag.iloc[start:stop][FRAG_COLUMNS].to_numpy()
        mz_block = mz_df.iloc[start:stop][FRAG_COLUMNS].to_numpy()
        for r in range(n_rows):
            for c, column in enumerate(FRAG_COLUMNS):
                if block[r, c] <= 0:
                    continue
                checked += 1
                ion_type, charge = column.split("_z")
                ordinal = (r + 1) if ion_type == "b" else (n_rows - r)
                found = peaks.get((ion_type, ordinal, int(charge)))
                if not found:
                    unmatched += 1
                    continue
                theoretical = mz_block[r, c]
                mz_obs, stated = min(found, key=lambda t: abs(t[0] - theoretical))
                ours.append((mz_obs - theoretical) / theoretical * 1e6)
                if stated is not None:
                    theirs.append((stated, ours[-1]))

    ours = np.array(ours)
    print(f"\nspectra compared          {len(observed)}")
    print(f"cells checked             {checked}")
    print(f"cells with no source peak {unmatched}")
    if len(ours) == 0:
        print("FAIL: nothing to compare", file=sys.stderr)
        return 1
    print(f"|observed - AlphaBase|    median {np.median(np.abs(ours)):.3f} ppm   "
          f"max {np.abs(ours).max():.3f} ppm")
    within = (np.abs(ours) < args.max_ppm).mean()
    print(f"within {args.max_ppm:g} ppm             {within:.2%}")

    ok = unmatched == 0 and within > 0.99
    if theirs:
        stated = np.array([t[0] for t in theirs])
        computed = np.array([t[1] for t in theirs])
        corr = float(np.corrcoef(stated, computed)[0, 1])
        delta = np.abs(computed - stated)
        print(f"\nagainst NIST's own stated ppm:")
        print(f"  correlation             {corr:.6f}")
        print(f"  |difference| median     {np.median(delta):.4f} ppm   max {delta.max():.4f} ppm")
        print(f"  agree within 1 ppm      {(delta < 1).mean():.2%}")
        ok = ok and corr > 0.99 and delta.max() < 5

    print("\nVERDICT:", "conversion CORRECT" if ok else "PROBLEM - do not train on this")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
