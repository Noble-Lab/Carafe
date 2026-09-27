#!/usr/bin/env python
"""Convert the CID spectra of PROSPECT PTMs: TMT into Carafe MS2 training data.

Reads the per-pool files downloaded from Zenodo (record 8221499) in --prospect-dir:
    <pool>_meta_data.parquet   one row per spectrum (raw file, scan, ProForma sequence, charge,
                               fragmentation, mass analyser, collision energy, Andromeda score)
    <pool>.zip                 <pool>/<raw file>_annotation.parquet, one row per annotated peak
and writes the two files Carafe's ai.py trains from:
    psm_pdv.txt                one row per spectrum (sequence, charge, mods, mod_sites, nce,
                               instrument, fragment row range, n_frags; plus raw_file,
                               scan_number and andromeda_score to trace each spectrum back)
    fragment_intensity_df.tsv  b_z1, b_z2, y_z1, y_z2 per backbone position (b_n at row n-1,
                               y_n at row L-n-1), each spectrum scaled to a maximum of 1
    conversion.json            counts, and what was dropped and why

Kept: CID spectra read in the ion trap (ITMS) whose peptide carries TMT6plex on the N-terminus and
on every lysine, with no modifications other than TMT6plex, carbamidomethyl C and oxidised M.

A peak can carry several annotations (for example y3, and y4 with a neutral loss, at the same
m/z). Each peak takes the one with the highest fragment_score, the first listed on a tie. A peak
whose reading is a b or y ion with no neutral loss at fragment charge 1 or 2 fills that ion's
cell; if several peaks fill one cell, the most intense is kept. Spectra with no such ion are
dropped. Raw files are converted in parallel and written in a fixed order, so the output does not
depend on the number of workers.

    python prospect_to_carafe_training.py --prospect-dir prospect --out-dir train_data
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

FRAG_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]
# UNIMOD id -> (Carafe/AlphaBase modification name, the residue it must sit on; None = N-terminus)
UNIMOD = {737: ("TMT6plex", "K"), 4: ("Carbamidomethyl", "C"), 35: ("Oxidation", "M")}
INSTRUMENT = "Lumos"  # AlphaPeptDeep instrument; PROSPECT does not record one per spectrum
META_COLUMNS = ["raw_file", "scan_number", "modified_sequence", "precursor_charge", "fragmentation",
                "mass_analyzer", "andromeda_score", "orig_collision_energy"]
ANNOTATION_COLUMNS = ["ion_type", "no", "charge", "experimental_mass", "intensity", "neutral_loss",
                      "fragment_score", "scan_number"]


def parse_proforma(modified: str):
    """ProForma (as PROSPECT writes it) -> (sequence, mods, mod_sites, fully TMT labelled).

    Raises ValueError for anything not in UNIMOD or on the wrong residue."""
    rest, mods, sites, nterm_label = modified, [], [], False
    match = re.match(r"^\[UNIMOD:(\d+)\]-", rest)
    if match:
        unimod = int(match.group(1))
        if unimod != 737:
            raise ValueError(f"unexpected N-terminal modification UNIMOD:{unimod}")
        mods.append("TMT6plex@Any N-term")
        sites.append(0)
        nterm_label = True
        rest = rest[match.end():]
    residues = re.findall(r"([A-Z])(?:\[UNIMOD:(\d+)\])?", rest)
    if "".join(aa + (f"[UNIMOD:{u}]" if u else "") for aa, u in residues) != rest:
        raise ValueError(f"cannot parse {modified!r}")
    sequence, all_k_labelled = "", True
    for position, (aa, unimod) in enumerate(residues, start=1):
        sequence += aa
        if unimod:
            name, residue = UNIMOD.get(int(unimod), (None, None))
            if name is None or residue != aa:
                raise ValueError(f"unexpected UNIMOD:{unimod} on {aa} in {modified!r}")
            mods.append(f"{name}@{aa}")
            sites.append(position)
        elif aa == "K":
            all_k_labelled = False
    return sequence, ";".join(mods), ";".join(str(s) for s in sites), nterm_label and all_k_labelled


def select_spectra(prospect_dir):
    """Metadata of the spectra to convert, and counts of what was left out."""
    frames = []
    for path in sorted(glob.glob(os.path.join(prospect_dir, "*_meta_data.parquet"))):
        meta = pd.read_parquet(path, columns=META_COLUMNS)
        meta["pool"] = os.path.basename(path)[:-len("_meta_data.parquet")]
        frames.append(meta)
    if not frames:
        raise SystemExit(f"no *_meta_data.parquet in {prospect_dir}")
    meta = pd.concat(frames, ignore_index=True)
    counts = {"spectra_in_metadata": len(meta)}
    cid = meta[(meta["fragmentation"] == "CID") & (meta["mass_analyzer"] == "ITMS")].copy()
    counts["cid_itms_spectra"] = len(cid)
    parsed = {}
    for modified in cid["modified_sequence"].unique():
        try:
            parsed[modified] = parse_proforma(modified)
        except ValueError:
            parsed[modified] = None
    info = cid["modified_sequence"].map(parsed)
    counts["dropped_unexpected_modification"] = int(info.isna().sum())
    cid = cid[info.notna()].copy()
    info = info[info.notna()]
    cid["sequence"] = [p[0] for p in info]
    cid["mods"] = [p[1] for p in info]
    cid["mod_sites"] = [p[2] for p in info]
    labelled = np.array([p[3] for p in info], dtype=bool)
    counts["dropped_not_fully_tmt_labelled"] = int((~labelled).sum())
    cid = cid[labelled]
    duplicated = cid.duplicated(["raw_file", "scan_number"], keep=False)
    counts["dropped_duplicate_scan"] = int(duplicated.sum())
    cid = cid[~duplicated]
    return cid.sort_values(["pool", "raw_file", "scan_number"]).reset_index(drop=True), counts


def convert_raw_file(job):
    """One raw file: (psm rows, fragment matrix, spectra without annotation, spectra without ions)."""
    prospect_dir, pool, raw_file, spectra = job
    with zipfile.ZipFile(os.path.join(prospect_dir, f"{pool}.zip")) as archive:
        member = f"{pool}/{raw_file}_annotation.parquet"
        ann = pd.read_parquet(io.BytesIO(archive.read(member)), columns=ANNOTATION_COLUMNS)
    ann = ann[ann["scan_number"].isin(spectra["scan_number"])]
    ann["listed"] = np.arange(len(ann))
    # one reading per peak: the highest fragment_score, the first listed on a tie
    ann = ann.sort_values(["scan_number", "experimental_mass", "fragment_score", "listed"],
                          ascending=[True, True, False, True])
    ann = ann.drop_duplicates(["scan_number", "experimental_mass"], keep="first")
    clean = ann[ann["ion_type"].isin(["b", "y"]) & (ann["neutral_loss"].fillna("") == "")
                & ann["charge"].between(1, 2)]

    spectra = spectra.reset_index(drop=True)
    annotated = spectra["scan_number"].isin(ann["scan_number"])
    n_rows = spectra["sequence"].str.len().to_numpy() - 1
    offsets = np.concatenate(([0], np.cumsum(n_rows)[:-1]))
    position = pd.Series(np.arange(len(spectra)), index=spectra["scan_number"].to_numpy())
    index = position.loc[clean["scan_number"].to_numpy()].to_numpy()
    ordinal = clean["no"].to_numpy()
    rows = n_rows[index]
    is_b = (clean["ion_type"] == "b").to_numpy()
    row = np.where(is_b, ordinal - 1, rows - ordinal)
    ok = (ordinal >= 1) & (ordinal <= rows) & (row >= 0) & (row < rows)
    column = np.where(is_b, 0, 2) + clean["charge"].to_numpy() - 1
    matrix = np.zeros((int(n_rows.sum()), len(FRAG_COLUMNS)), dtype=np.float32)
    np.maximum.at(matrix, (offsets[index[ok]] + row[ok], column[ok]),
                  clean["intensity"].to_numpy(dtype=np.float32)[ok])

    n_frags = np.add.reduceat((matrix > 0).sum(axis=1), offsets) if len(spectra) else np.array([])
    peak = np.maximum.reduceat(matrix.max(axis=1), offsets) if len(spectra) else np.array([])
    keep = annotated.to_numpy() & (n_frags > 0)
    blocks = [matrix[offsets[i]:offsets[i] + n_rows[i]] / peak[i] for i in np.flatnonzero(keep)]
    kept = spectra[keep].copy()
    kept["n_frags"] = n_frags[keep].astype(int)
    frag = np.concatenate(blocks) if blocks else np.zeros((0, len(FRAG_COLUMNS)), dtype=np.float32)
    return kept, frag, int((~annotated).sum()), int((annotated.to_numpy() & (n_frags == 0)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prospect-dir", required=True, help="directory with the PROSPECT TMT files from Zenodo")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    start = time.time()
    spectra, counts = select_spectra(args.prospect_dir)
    print(f"{len(spectra):,} spectra selected from {counts['cid_itms_spectra']:,} CID/ITMS spectra", flush=True)
    jobs = [(args.prospect_dir, pool, raw_file, group[["scan_number", "sequence", "precursor_charge", "mods",
                                                       "mod_sites", "orig_collision_energy", "andromeda_score"]])
            for (pool, raw_file), group in spectra.groupby(["pool", "raw_file"], sort=True)]

    os.makedirs(args.out_dir, exist_ok=True)
    psm_path = os.path.join(args.out_dir, "psm_pdv.txt")
    frag_path = os.path.join(args.out_dir, "fragment_intensity_df.tsv")
    stats = Counter()
    charge, nce, pools = Counter(), Counter(), Counter()
    cursor = written = 0
    with open(psm_path + ".part", "w", newline="") as psm_out, open(frag_path + ".part", "w", newline="") as frag_out:
        psm_out.write("sequence\tcharge\tmods\tmod_sites\tnce\tinstrument\tfrag_start_idx\tfrag_stop_idx\t"
                      "n_frags\traw_file\tscan_number\tandromeda_score\n")
        frag_out.write("\t".join(FRAG_COLUMNS) + "\n")
        with Pool(args.workers) as pool:
            for k, (job, (kept, frag, no_annotation, no_ions)) in enumerate(
                    zip(jobs, pool.imap(convert_raw_file, jobs, chunksize=1)), start=1):
                stats["dropped_no_annotation"] += no_annotation
                stats["dropped_no_clean_by_ion"] += no_ions
                n_rows = kept["sequence"].str.len().to_numpy() - 1
                stops = cursor + np.cumsum(n_rows)
                table = pd.DataFrame({
                    "sequence": kept["sequence"], "charge": kept["precursor_charge"], "mods": kept["mods"],
                    "mod_sites": kept["mod_sites"], "nce": kept["orig_collision_energy"].map("{:g}".format),
                    "instrument": INSTRUMENT, "frag_start_idx": stops - n_rows, "frag_stop_idx": stops,
                    "n_frags": kept["n_frags"], "raw_file": job[2], "scan_number": kept["scan_number"],
                    "andromeda_score": kept["andromeda_score"]})
                table.to_csv(psm_out, sep="\t", index=False, header=False)
                np.savetxt(frag_out, frag, fmt="%.6g", delimiter="\t")
                cursor += len(frag)
                written += len(kept)
                charge.update(kept["precursor_charge"].tolist())
                nce.update(table["nce"].tolist())
                pools[job[1]] += len(kept)
                if k % 100 == 0 or k == len(jobs):
                    print(f"{k} of {len(jobs)} raw files, {written:,} spectra ({time.time() - start:.0f} s)", flush=True)
    os.replace(psm_path + ".part", psm_path)
    os.replace(frag_path + ".part", frag_path)

    summary = {**counts, "selected": len(spectra), **dict(stats), "spectra": written, "fragment_rows": cursor,
               "raw_files": len(jobs), "charge": {str(k): v for k, v in sorted(charge.items())},
               "nce": dict(nce), "pools": dict(sorted(pools.items())), "instrument": INSTRUMENT}
    with open(os.path.join(args.out_dir, "conversion.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {written:,} spectra and {cursor:,} fragment rows to {args.out_dir} ({time.time() - start:.0f} s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
