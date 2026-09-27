#!/usr/bin/env python
"""Unit tests for the TMT CID model scripts. Run: python -m pytest test_ms2cid_tools.py"""
from __future__ import annotations

import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prospect_to_carafe_training import convert_raw_file, parse_proforma  # noqa: E402
from split_prosit_holdout import split  # noqa: E402
import plot_loss_comparison  # noqa: E402


# ---- parse_proforma ---------------------------------------------------------------------------

def test_parse_labelled_peptide():
    assert parse_proforma("[UNIMOD:737]-PEPC[UNIMOD:4]M[UNIMOD:35]K[UNIMOD:737]") == \
        ("PEPCMK", "TMT6plex@Any N-term;Carbamidomethyl@C;Oxidation@M;TMT6plex@K", "0;4;5;6", True)


@pytest.mark.parametrize("modified", ["PEPTIDEK",                         # no label at all
                                      "[UNIMOD:737]-PEPKTIDEK[UNIMOD:737]",  # one lysine unlabelled
                                      "PEPK[UNIMOD:737]"])                   # N-terminus unlabelled
def test_not_fully_labelled(modified):
    assert parse_proforma(modified)[3] is False


@pytest.mark.parametrize("modified", ["[UNIMOD:737]-PEPS[UNIMOD:21]K[UNIMOD:737]",  # modification not handled
                                      "[UNIMOD:737]-PEPC[UNIMOD:35]K[UNIMOD:737]",  # oxidation on C
                                      "[UNIMOD:4]-PEPK[UNIMOD:737]"])               # wrong N-terminal label
def test_rejects_unexpected_modifications(modified):
    with pytest.raises(ValueError):
        parse_proforma(modified)


# ---- convert_raw_file -------------------------------------------------------------------------

def make_pool(tmp_path, annotations):
    pool, raw = "TMT_TEST_pool", "run1"
    path = tmp_path / "ann.parquet"
    annotations.to_parquet(path)
    with zipfile.ZipFile(tmp_path / f"{pool}.zip", "w") as archive:
        archive.write(path, f"{pool}/{raw}_annotation.parquet")
    return pool, raw


def test_convert_follows_the_rules(tmp_path):
    # AAAK (4 residues, 3 rows). Peak at 300 has two readings: y1 (score 100) wins over b2-H2O (score 70).
    # b1 is observed twice (two peaks): the stronger one counts. y2^3 and the precursor are left out.
    rows = [
        ("y", 1, 1, 300.0, 0.5, "", 100), ("b", 2, 1, 300.0, 0.5, "H2O", 70),
        ("b", 1, 1, 200.0, 0.2, "", 100), ("b", 1, 1, 200.4, 0.8, "", 90),
        ("b", 2, 2, 150.0, 1.0, "", 100), ("y", 2, 3, 120.0, 2.0, "", 100),
        ("precursor", 0, 2, 600.0, 3.0, "", 100),
    ]
    ann = pd.DataFrame(rows, columns=["ion_type", "no", "charge", "experimental_mass", "intensity",
                                      "neutral_loss", "fragment_score"])
    ann["scan_number"] = 7
    pool, raw = make_pool(tmp_path, ann)
    spectra = pd.DataFrame({"scan_number": [7, 8], "sequence": ["AAAK", "AAAK"], "precursor_charge": [2, 2],
                            "mods": ["", ""], "mod_sites": ["", ""], "orig_collision_energy": [35.0, 35.0],
                            "andromeda_score": [100.0, 90.0]})
    kept, frag, no_annotation, no_ions = convert_raw_file((str(tmp_path), pool, raw, spectra))
    assert (no_annotation, no_ions) == (1, 0)                   # scan 8 has no annotations
    assert list(kept["scan_number"]) == [7] and int(kept["n_frags"].iloc[0]) == 3
    expected = np.zeros((3, 4), dtype=np.float32)               # columns b_z1, b_z2, y_z1, y_z2
    expected[0, 0] = 0.8                                        # b1: the stronger of its two peaks
    expected[1, 1] = 1.0                                        # b2^2
    expected[2, 2] = 0.5                                        # y1 at row L-1-1 = 2
    assert np.allclose(frag, expected)                          # (already scaled: the maximum is 1.0)


# ---- split_prosit_holdout ---------------------------------------------------------------------

def test_split_holds_out_prosit_hold_out_and_keeps_sequences_apart():
    sequences = [f"PEPTIDE{c}K" for c in "ACDEFGHIKLMNPQRSTVWY"] * 3
    psm = pd.DataFrame({"sequence": sequences})
    lengths = psm["sequence"].str.len().to_numpy() - 1
    stops = np.cumsum(lengths)
    psm["frag_start_idx"], psm["frag_stop_idx"] = stops - lengths, stops
    frag = pd.DataFrame(np.arange(stops[-1] * 4, dtype=float).reshape(-1, 4), columns=["b_z1", "b_z2", "y_z1", "y_z2"])
    prosit = {"PEPTIDEAK": "hold_out", "PEPTIDECK": "train", "PEPTIDEDK": "validation"}
    parts, counts = split(psm, frag, prosit, valid_fraction=0.2, seed=1)
    held = set(parts["heldout"][0]["sequence"])
    assert held == {"PEPTIDEAK"}
    sets = [set(p[0]["sequence"]) for p in parts.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(p[0]) for p in parts.values()) == len(psm)
    assert counts["valid_sequences"] == 4                        # 20% of the 19 non-held-out sequences


# ---- plot_loss_comparison ---------------------------------------------------------------------

def _history(run_dir, lr, scale):
    os.makedirs(run_dir)
    epochs = np.arange(1, len(lr) + 1)
    pd.DataFrame({"epoch": epochs, "lr": lr, "train_loss": scale / epochs,
                  "test_loss": scale * 1.1 / epochs}).to_csv(os.path.join(run_dir, "ms2_history.tsv"),
                                                              sep="	", index=False)


def test_comparison_figure_is_written(tmp_path):
    lr = [1e-4, 2e-4, 1e-4, 0.0]
    _history(str(tmp_path / "G"), lr, 0.05)
    _history(str(tmp_path / "H"), lr, 0.15)
    out = str(tmp_path / "curve")
    plot_loss_comparison.main(["--run", f"G={tmp_path / 'G'}=the HCD TMT model",
                               "--run", f"H={tmp_path / 'H'}=random weights", "--out", out, "--title", "t"])
    for mode in ("light", "dark"):
        assert os.path.getsize(f"{out}_{mode}.png") > 0


def test_comparison_refuses_different_schedules(tmp_path):
    _history(str(tmp_path / "G"), [1e-4, 2e-4, 0.0], 0.05)
    _history(str(tmp_path / "H"), [1e-4, 1e-4, 0.0], 0.15)
    with pytest.raises(SystemExit):
        plot_loss_comparison.main(["--run", f"G={tmp_path / 'G'}", "--run", f"H={tmp_path / 'H'}",
                                   "--out", str(tmp_path / "curve"), "--title", "t"])
