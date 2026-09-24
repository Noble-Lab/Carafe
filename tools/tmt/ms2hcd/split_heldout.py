#!/usr/bin/env python
"""Split MS2 training data into a training part and a fixed held-out test set, by peptide sequence.

The held-out set is for the final, one-time evaluation of a chosen model (evaluate_heldout.py) and
is never used to pick anything. ai.py keeps its own internal split of the training part for early
stopping and best-epoch selection; reporting that set's metrics as the result would count the same
spectra twice, once to choose and once to score.

Every spectrum of a held-out sequence goes to the held-out set (whatever its charge, modifications,
NCE or instrument), so no test peptide is seen in training. Sequences are chosen with a fixed seed.

Input and output directories hold the two files ai.py reads:
    psm_pdv.txt                one row per spectrum; frag_start_idx/frag_stop_idx point into
    fragment_intensity_df.tsv  the fragment rows (one per backbone position)
The fragment row ranges are rebuilt for each output so that they tile its fragment table.

    python split_heldout.py --in-dir train_data --out-dir split --fraction 0.05 --seed 2024
writes split/train/, split/heldout/ and split/split.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def fragment_rows(psm: pd.DataFrame) -> np.ndarray:
    """Indices of the fragment rows belonging to these PSMs, in PSM order."""
    starts = psm["frag_start_idx"].to_numpy(dtype=np.int64)
    lengths = psm["frag_stop_idx"].to_numpy(dtype=np.int64) - starts
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
    return np.arange(lengths.sum(), dtype=np.int64) + offsets


def reindex(psm: pd.DataFrame, fragments: pd.DataFrame):
    """Copy of the PSMs and their fragment rows, with fragment ranges rebuilt from 0."""
    rows = fragment_rows(psm)
    out_frag = fragments.iloc[rows].reset_index(drop=True)
    lengths = (psm["frag_stop_idx"] - psm["frag_start_idx"]).to_numpy(dtype=np.int64)
    stops = np.cumsum(lengths)
    out_psm = psm.copy()
    out_psm["frag_start_idx"] = stops - lengths
    out_psm["frag_stop_idx"] = stops
    return out_psm.reset_index(drop=True), out_frag


# Carafe's default library range (-minLength 7 -maxLength 35 -min_pep_charge 2 -max_pep_charge 4).
DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH = 7, 35
DEFAULT_MIN_CHARGE, DEFAULT_MAX_CHARGE = 2, 4


def in_default_range(psm: pd.DataFrame) -> np.ndarray:
    """True for spectra inside Carafe's default library range of length and precursor charge."""
    length = psm["sequence"].str.len()
    return ((length >= DEFAULT_MIN_LENGTH) & (length <= DEFAULT_MAX_LENGTH)
            & (psm["charge"] >= DEFAULT_MIN_CHARGE) & (psm["charge"] <= DEFAULT_MAX_CHARGE)).to_numpy()


def restrict_to_default_range(in_dir: str, out_dir: str):
    """Write the default-range spectra of in_dir (psm_pdv.txt + fragment_intensity_df.tsv) to out_dir."""
    psm = pd.read_csv(os.path.join(in_dir, "psm_pdv.txt"), sep="\t",
                      dtype={"mods": str, "mod_sites": str}, keep_default_na=False)
    fragments = pd.read_csv(os.path.join(in_dir, "fragment_intensity_df.tsv"), sep="\t")
    out_psm, out_frag = reindex(psm[in_default_range(psm)], fragments)
    write(out_dir, out_psm, out_frag)
    return len(out_psm), len(psm)


def split(psm: pd.DataFrame, fragments: pd.DataFrame, fraction: float, seed: int):
    sequences = np.sort(psm["sequence"].unique())
    rng = np.random.default_rng(seed)
    n_heldout = int(round(len(sequences) * fraction))
    heldout_sequences = set(rng.choice(sequences, size=n_heldout, replace=False))
    is_heldout = psm["sequence"].isin(heldout_sequences).to_numpy()
    return reindex(psm[~is_heldout], fragments), reindex(psm[is_heldout], fragments), len(sequences), n_heldout


def write(directory: str, psm: pd.DataFrame, fragments: pd.DataFrame):
    os.makedirs(directory, exist_ok=True)
    psm.to_csv(os.path.join(directory, "psm_pdv.txt"), sep="\t", index=False)
    fragments.to_csv(os.path.join(directory, "fragment_intensity_df.tsv"), sep="\t", index=False,
                     float_format="%.6g")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="output of msp_to_carafe_training.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fraction", type=float, default=0.05, help="share of peptide sequences held out")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)

    psm = pd.read_csv(os.path.join(args.in_dir, "psm_pdv.txt"), sep="\t",
                      dtype={"mods": str, "mod_sites": str}, keep_default_na=False)
    fragments = pd.read_csv(os.path.join(args.in_dir, "fragment_intensity_df.tsv"), sep="\t")
    (train_psm, train_frag), (test_psm, test_frag), n_seq, n_held = split(psm, fragments, args.fraction, args.seed)

    if set(train_psm["sequence"]) & set(test_psm["sequence"]):
        sys.exit("internal error: a sequence is in both parts")
    if len(train_frag) + len(test_frag) != len(fragments):
        sys.exit("internal error: fragment rows lost or duplicated")

    write(os.path.join(args.out_dir, "train"), train_psm, train_frag)
    write(os.path.join(args.out_dir, "heldout"), test_psm, test_frag)
    summary = {
        "fraction": args.fraction,
        "seed": args.seed,
        "sequences": n_seq,
        "heldout_sequences": n_held,
        "train_spectra": len(train_psm),
        "heldout_spectra": len(test_psm),
        "train_fragment_rows": len(train_frag),
        "heldout_fragment_rows": len(test_frag),
    }
    with open(os.path.join(args.out_dir, "split.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
