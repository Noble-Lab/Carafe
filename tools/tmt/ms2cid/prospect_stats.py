#!/usr/bin/env python
"""Describe PROSPECT PTMs: TMT, and how its CID spectra compare with the Prosit-TMT SI.

Reads the per-pool metadata files (one row per spectrum) and, if given, Supplementary Table 1 of
Gabriel et al., Anal. Chem. 2022 (Prosit-TMT). Writes a JSON summary and prints it.

    python prospect_stats.py --prospect-dir prospect --si ac1c05435_si_001.xlsx --out prospect_stats.json

"Fully labelled" means TMT6plex on the peptide N-terminus and on every lysine.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

COLUMNS = ["raw_file", "scan_number", "modified_sequence", "precursor_charge", "fragmentation", "mass_analyzer",
           "andromeda_score", "peptide_length", "orig_collision_energy"]


def load(prospect_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(prospect_dir, "*_meta_data.parquet"))):
        meta = pd.read_parquet(path, columns=COLUMNS)
        meta["pool"] = os.path.basename(path)[:-len("_meta_data.parquet")]
        frames.append(meta)
    m = pd.concat(frames, ignore_index=True)
    seq = m["modified_sequence"]
    m["unmod"] = seq.str.replace(r"\[UNIMOD:\d+\]-?", "", regex=True)
    nterm = seq.str.startswith("[UNIMOD:737]-")
    any_tmt = seq.str.contains("UNIMOD:737", regex=False)
    all_k = seq.str.count(r"K(?!\[UNIMOD:737\])") == 0
    m["label"] = "none"
    m.loc[any_tmt, "label"] = "partial"
    m.loc[nterm & all_k, "label"] = "full"
    return m


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def summarize(m, si_path=None):
    cid = m[(m["fragmentation"] == "CID") & (m["mass_analyzer"] == "ITMS")]
    full = cid[cid["label"] == "full"]
    per_ion = full.groupby(["modified_sequence", "precursor_charge"]).size()
    out = {
        "all": {
            "spectra": len(m), "raw_files": int(m["raw_file"].nunique()), "pools": sorted(m["pool"].unique()),
            "fragmentation_x_analyser": {f"{f}/{a}": int(n) for (f, a), n in
                                         m.groupby(["fragmentation", "mass_analyzer"]).size().items()},
            "label": counts(m["label"]),
            "hcd_ftms_collision_energy": counts(m.loc[(m["fragmentation"] == "HCD") & (m["mass_analyzer"] == "FTMS"),
                                                      "orig_collision_energy"]),
        },
        "cid_itms": {
            "spectra": len(cid), "label": counts(cid["label"]),
            "collision_energy": counts(cid["orig_collision_energy"]),
            "raw_file_method": counts(cid["raw_file"].str.extract(r"-([^-]+)-1h-R\d")[0]),
        },
        "cid_itms_fully_labelled": {
            "spectra": len(full), "sequences": int(full["unmod"].nunique()),
            "modified_peptides": int(full["modified_sequence"].nunique()), "peptide_ions": int(len(per_ion)),
            "precursor_charge": counts(full["precursor_charge"]),
            "length_min": int(full["peptide_length"].min()), "length_max": int(full["peptide_length"].max()),
            "length_above_30": int((full["peptide_length"] > 30).sum()),
            "spectra_per_peptide_ion": {"median": float(per_ion.median()), "mean": round(float(per_ion.mean()), 2),
                                        "max": int(per_ion.max())},
            "andromeda_score_min": float(full["andromeda_score"].min()),
            "andromeda_score_median": float(full["andromeda_score"].median()),
            "with_oxidation": int(full["modified_sequence"].str.contains("UNIMOD:35]", regex=False).sum()),
            "per_pool": {p: {"spectra": int(len(g)), "sequences": int(g["unmod"].nunique())}
                         for p, g in full.groupby("pool")},
        },
    }
    if si_path:
        obs = pd.read_excel(si_path, sheet_name="PT TMT labeled peptides observe")
        split = pd.read_excel(si_path, sheet_name="Train-Validation-Holdout")
        si_cid = obs[(obs["fragmentation"] == "CID") & (obs["mass analyser"] == "ITMS")]
        split_of = dict(zip(split["sequence"], split["dataset"]))
        pro_cid = set(full["unmod"])
        si_cid_seqs = set(si_cid["unmod sequence"])
        present = set(zip(m["raw_file"], m["scan_number"]))
        out["prosit_tmt_si"] = {
            "observed_rows": len(obs),
            "fragmentation_x_analyser": {f"{f}/{a}": int(n) for (f, a), n in
                                         obs.groupby(["fragmentation", "mass analyser"]).size().items()},
            "split_sequences": counts(split["dataset"]),
            "cid_rows_sequences": len(si_cid_seqs),
            "cid_sequences_shared_with_prospect": len(pro_cid & si_cid_seqs),
            "prospect_cid_sequences_by_split": counts(pd.Series([split_of.get(s, "not in split") for s in pro_cid])),
            "representative_scans_in_prospect": int(sum(k in present for k in zip(obs["rawfile"], obs["scan number"]))),
            "cid_representative_scans_in_prospect": int(sum(k in present for k in
                                                            zip(si_cid["rawfile"], si_cid["scan number"]))),
            "raw_files": int(obs["rawfile"].nunique()),
            "raw_files_shared": len(set(m["raw_file"]) & set(obs["rawfile"])),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prospect-dir", required=True)
    ap.add_argument("--si", help="Prosit-TMT Supplementary Table 1 (ac1c05435_si_001.xlsx)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summary = summarize(load(args.prospect_dir), args.si)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
