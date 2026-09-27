#!/usr/bin/env python
"""Split CID training data into training, validation and held-out parts, by peptide sequence.

The held-out test set is every spectrum whose (unmodified) sequence is labelled hold_out in
Supplementary Table 1 of Gabriel et al., Anal. Chem. 2022 (Prosit-TMT), sheet
'Train-Validation-Holdout'. None of those sequences is in the training or validation parts, so the
Carafe models are scored on peptides they did not see.

A fraction of the remaining sequences (seed 2024) is set aside as the validation set that ai.py
uses to choose the best epoch (train_tmt_ms2.py --valid-dir); everything else is training data.
The held-out set is never used to choose anything.

    python split_prosit_holdout.py --in-dir train_data --si ac1c05435_si_001.xlsx --out-dir split
writes split/train/, split/valid/, split/heldout/ and split/split.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ms2hcd"))
from split_heldout import reindex, write  # noqa: E402

SI_SHEET = "Train-Validation-Holdout"


def read_prosit_split(si_path):
    table = pd.read_excel(si_path, sheet_name=SI_SHEET)
    return dict(zip(table["sequence"], table["dataset"]))


def split(psm, fragments, prosit_split, valid_fraction, seed):
    prosit_set = psm["sequence"].map(prosit_split).fillna("not in split")
    held = (prosit_set == "hold_out").to_numpy()
    rest = np.sort(psm.loc[~held, "sequence"].unique())
    rng = np.random.default_rng(seed)
    valid_sequences = set(rng.choice(rest, size=int(round(len(rest) * valid_fraction)), replace=False))
    valid = psm["sequence"].isin(valid_sequences).to_numpy() & ~held
    train = ~held & ~valid
    parts = {name: reindex(psm[mask], fragments) for name, mask in
             (("train", train), ("valid", valid), ("heldout", held))}
    counts = {
        "sequences": int(psm["sequence"].nunique()),
        "prosit_set_of_spectra": {k: int(v) for k, v in prosit_set.value_counts().items()},
        "prosit_set_of_training_spectra": {k: int(v) for k, v in prosit_set[train].value_counts().items()},
        **{f"{name}_spectra": int(len(p[0])) for name, p in parts.items()},
        **{f"{name}_sequences": int(p[0]["sequence"].nunique()) for name, p in parts.items()},
        **{f"{name}_fragment_rows": int(len(p[1])) for name, p in parts.items()},
    }
    return parts, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="output of prospect_to_carafe_training.py")
    ap.add_argument("--si", required=True, help="Prosit-TMT Supplementary Table 1 (ac1c05435_si_001.xlsx)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--valid-fraction", type=float, default=0.01, help="share of non-held-out sequences")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)

    psm = pd.read_csv(os.path.join(args.in_dir, "psm_pdv.txt"), sep="\t",
                      dtype={"mods": str, "mod_sites": str}, keep_default_na=False)
    fragments = pd.read_csv(os.path.join(args.in_dir, "fragment_intensity_df.tsv"), sep="\t")
    parts, counts = split(psm, fragments, read_prosit_split(args.si), args.valid_fraction, args.seed)

    names = [set(p[0]["sequence"]) for p in parts.values()]
    if names[0] & names[1] or names[0] & names[2] or names[1] & names[2]:
        sys.exit("internal error: a sequence is in two parts")
    if sum(len(p[1]) for p in parts.values()) != len(fragments):
        sys.exit("internal error: fragment rows lost or duplicated")
    for name, (part_psm, part_frag) in parts.items():
        write(os.path.join(args.out_dir, name), part_psm, part_frag)
    summary = {"valid_fraction": args.valid_fraction, "seed": args.seed, **counts}
    with open(os.path.join(args.out_dir, "split.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
