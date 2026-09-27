#!/usr/bin/env python
"""Check that prospect_to_carafe_training.py put every intensity in the right fragment cell.

A wrong sequence, modification position, fragment row or charge column would still give a
plausible-looking file, so a random sample of converted spectra is checked from the outside:

1. Layout: the fragment blocks tile the fragment table, one row per backbone position.
2. Recomputation: each sampled spectrum is rebuilt from its PROSPECT annotations with a separate,
   deliberately plain implementation of the same rules, and must equal the converted rows.
3. Masses: for every populated cell, AlphaBase's theoretical m/z for that cell (from the sequence
   and modifications written to psm_pdv.txt) must equal the theoretical m/z PROSPECT stored for
   the annotation the intensity came from. A wrong modification position would shift a
   TMT-bearing fragment by 229 Da. The observed peak's deviation is reported as well.

    python verify_prospect_conversion.py --prospect-dir prospect --train-dir train_data
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

FRAG_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]


def expected_block(annotations: pd.DataFrame, length: int):
    """The converter's rules, written out plainly: {(ion, ordinal, charge): (intensity, theoretical m/z)}."""
    best = {}  # peak m/z -> (score, listing order, annotation)
    for order, a in enumerate(annotations.itertuples(index=False)):
        key = a.experimental_mass
        if key not in best or a.fragment_score > best[key][0]:
            best[key] = (a.fragment_score, order, a)
    cells = {}
    for _, _, a in best.values():
        loss = a.neutral_loss if isinstance(a.neutral_loss, str) else ""
        if a.ion_type in ("b", "y") and loss == "" and 1 <= a.charge <= 2 and 1 <= a.no <= length - 1:
            cell = (a.ion_type, int(a.no), int(a.charge))
            if cell not in cells or a.intensity > cells[cell][0]:
                cells[cell] = (a.intensity, a.theoretical_mass, a.experimental_mass)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prospect-dir", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--sample", type=int, default=2000, help="spectra to check (default 2000)")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)
    import alphabase.peptide.fragment as fragment

    psm = pd.read_csv(os.path.join(args.train_dir, "psm_pdv.txt"), sep="\t",
                      dtype={"mods": str, "mod_sites": str}, keep_default_na=False)
    frag = pd.read_csv(os.path.join(args.train_dir, "fragment_intensity_df.tsv"), sep="\t")
    psm["nAA"] = psm["sequence"].str.len()

    starts, stops = psm["frag_start_idx"].to_numpy(), psm["frag_stop_idx"].to_numpy()
    if not ((stops - starts == psm["nAA"].to_numpy() - 1).all() and starts[0] == 0
            and (starts[1:] == stops[:-1]).all() and stops[-1] == len(frag)):
        print("FAIL: fragment blocks do not tile the fragment table", file=sys.stderr)
        return 1
    print(f"block layout OK: {len(psm):,} spectra tile {len(frag):,} fragment rows contiguously")

    sample = psm.sample(min(args.sample, len(psm)), random_state=args.seed).sort_index()
    mz = fragment.create_fragment_mz_dataframe(
        sample[["sequence", "mods", "mod_sites", "charge", "nAA"]].reset_index(drop=True).copy(), FRAG_COLUMNS)
    member_of = {}  # raw file -> (zip path, member name)
    for name in sorted(os.listdir(args.prospect_dir)):
        if name.endswith(".zip"):
            path = os.path.join(args.prospect_dir, name)
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.endswith("_annotation.parquet"):
                        member_of[os.path.basename(member)[:-len("_annotation.parquet")]] = (path, member)

    mismatched, cells, theo_ppm, obs_da, missing = 0, 0, [], [], 0
    mz_cursor = 0
    for raw_file, group in sample.groupby("raw_file", sort=False):
        path, member = member_of[raw_file]
        with zipfile.ZipFile(path) as archive:
            ann = pd.read_parquet(io.BytesIO(archive.read(member)))
        ann = ann[ann["scan_number"].isin(group["scan_number"])]
        by_scan = dict(tuple(ann.groupby("scan_number", sort=False)))
        for row in group.itertuples():
            n_rows = row.nAA - 1
            block = frag.iloc[row.frag_start_idx:row.frag_stop_idx][FRAG_COLUMNS].to_numpy()
            expected = expected_block(by_scan.get(row.scan_number, ann.iloc[:0]), row.nAA)
            top = max(v[0] for v in expected.values()) if expected else 0
            rebuilt = np.zeros_like(block)
            for (ion, ordinal, z), (intensity, theoretical, experimental) in expected.items():
                r = ordinal - 1 if ion == "b" else n_rows - ordinal
                c = FRAG_COLUMNS.index(f"{ion}_z{z}")
                rebuilt[r, c] = intensity / top
                ours = mz.iloc[mz_cursor + r][FRAG_COLUMNS[c]]
                theo_ppm.append(abs(ours - theoretical) / theoretical * 1e6)
                obs_da.append(experimental - ours)
                cells += 1
            missing += int(((block > 0) & (rebuilt == 0)).sum())
            if not np.allclose(block, rebuilt, rtol=1e-5, atol=1e-6):
                mismatched += 1
            mz_cursor += n_rows

    theo_ppm, obs_da = np.array(theo_ppm), np.array(obs_da)
    print(f"spectra checked                   {len(sample):,} from {sample['raw_file'].nunique()} raw files")
    print(f"cells checked                     {cells:,}")
    print(f"spectra differing from rebuild    {mismatched}")
    print(f"cells with no source annotation   {missing}")
    print(f"AlphaBase vs PROSPECT theoretical median {np.median(theo_ppm):.4f} ppm, max {theo_ppm.max():.4f} ppm")
    print(f"observed - AlphaBase m/z          median {np.median(obs_da):+.3f} Da, 99% within "
          f"{np.quantile(np.abs(obs_da), 0.99):.3f} Da (ion trap)")
    ok = mismatched == 0 and missing == 0 and cells > 0 and theo_ppm.max() < 1
    print("\nVERDICT:", "conversion CORRECT" if ok else "PROBLEM - do not train on this")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
