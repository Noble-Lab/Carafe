#!/usr/bin/env python
"""Unit tests for the NIST CPTAC3 TMT (HCD) model scripts. Run: python -m pytest test_ms2hcd_tools.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from library_stats import instrument_from_filename, instrument_from_resolution  # noqa: E402
from msp_to_carafe_training import parse_mods  # noqa: E402
from split_heldout import fragment_rows, in_default_range, reindex, restrict_to_default_range, split  # noqa: E402
from compare_prosit_tmt import in_prosit_range, proforma, to_fragment_rows  # noqa: E402


# ---- msp_to_carafe_training.parse_mods -------------------------------------------------------

def test_nterm_label_and_lysine():
    mods, sites = parse_mods("2(0,A,TMT6plex)(3,K,TMT6plex)", "AAAK")
    assert mods == "TMT6plex@Any N-term;TMT6plex@K"
    assert sites == "0;4"  # MSP positions are 0-based; mod_sites are 1-based


@pytest.mark.parametrize("residue1, second, expected", [
    ("K", "TMT6plex", "TMT6plex@K"),         # N-term label plus a labelled lysine at residue 1
    ("C", "CAM", "Carbamidomethyl@C"),       # N-term label plus CAM on a cysteine at residue 1
    ("M", "Oxidation", "Oxidation@M"),       # N-term label plus an oxidised methionine at residue 1
])
def test_two_position0_entries(residue1, second, expected):
    # The library writes both at position 0, the label first; the second belongs to residue 1.
    mods, sites = parse_mods(f"2(0,{residue1},TMT6plex)(0,{residue1},{second})", residue1 + "AAR")
    assert mods == f"TMT6plex@Any N-term;{expected}"
    assert sites == "0;1"


def test_position0_does_not_depend_on_order():
    # The label is the N-terminus wherever it is listed; CAM stays on the cysteine.
    assert parse_mods("2(0,C,CAM)(0,C,TMT6plex)", "CAAR") == ("TMT6plex@Any N-term;Carbamidomethyl@C", "0;1")


def test_position0_without_a_label_is_residue1():
    # A library without N-terminal labels: CAM at position 0 is on residue 1, not the N-terminus.
    assert parse_mods("1(0,C,CAM)", "CAAR") == ("Carbamidomethyl@C", "1")
    assert parse_mods("1(0,M,Oxidation)", "MAAR") == ("Oxidation@M", "1")


def test_rejects_residue_mismatch_and_unknown_names():
    with pytest.raises(ValueError):
        parse_mods("1(2,K,TMT6plex)", "AAAR")      # position 2 is A, not K
    with pytest.raises(ValueError):
        parse_mods("1(0,A,Phospho)", "AAAR")       # not a modification this library contains


# ---- library_stats instrument rules -----------------------------------------------------------

def test_instrument_rules():
    assert instrument_from_resolution(50000) == "Lumos"
    assert instrument_from_resolution(35000) == "QE"
    assert instrument_from_filename("CPTAC3_X_LUMOS_23.raw") == "Lumos"
    assert instrument_from_filename("CPTAC3_X_QE_f18.raw") == "QE"
    assert instrument_from_filename("CPTAC3_X_f18.raw") is None
    assert instrument_from_filename("CPTAC3_QEXACTIVE.raw") is None  # only a whole QE token counts


# ---- split_heldout ----------------------------------------------------------------------------

def toy_data():
    """Five spectra over three sequences; fragment row values encode (spectrum, position)."""
    psm = pd.DataFrame({
        "sequence": ["AAK", "PEPTIDER", "AAK", "LLK", "PEPTIDER"],
        "charge": [2, 2, 3, 2, 3],
    })
    lengths = [len(s) - 1 for s in psm["sequence"]]
    stops = np.cumsum(lengths)
    psm["frag_start_idx"] = stops - lengths
    psm["frag_stop_idx"] = stops
    values = [i * 100 + j for i, n in enumerate(lengths) for j in range(n)]
    frag = pd.DataFrame({"b_z1": values, "b_z2": 0.0, "y_z1": 0.0, "y_z2": 0.0})
    return psm, frag


def test_fragment_rows_follow_each_spectrum():
    psm, _ = toy_data()
    rows = fragment_rows(psm.iloc[[3, 1]])
    # rows per spectrum: 2, 7, 2, 2, 7 -> LLK (spectrum 3) is 11-12, PEPTIDER (spectrum 1) is 2-8
    assert list(rows) == list(range(11, 13)) + list(range(2, 9))


def test_reindex_keeps_each_spectrum_with_its_own_fragments():
    psm, frag = toy_data()
    out_psm, out_frag = reindex(psm.iloc[[4, 0]], frag)
    assert list(out_psm["frag_start_idx"]) == [0, 7]
    assert list(out_psm["frag_stop_idx"]) == [7, 9]
    assert list(out_frag["b_z1"][:7]) == [400 + j for j in range(7)]  # spectrum 4
    assert list(out_frag["b_z1"][7:]) == [0, 1]                        # spectrum 0


def test_split_is_sequence_disjoint_and_complete():
    psm, frag = toy_data()
    (tr_psm, tr_frag), (te_psm, te_frag), n_seq, n_held = split(psm, frag, fraction=0.34, seed=1)
    assert n_seq == 3 and n_held == 1
    assert not set(tr_psm["sequence"]) & set(te_psm["sequence"])
    assert len(tr_psm) + len(te_psm) == len(psm)
    assert len(tr_frag) + len(te_frag) == len(frag)
    for part_psm, part_frag in ((tr_psm, tr_frag), (te_psm, te_frag)):
        # contiguous tiling from 0
        assert list(part_psm["frag_start_idx"][1:]) == list(part_psm["frag_stop_idx"][:-1])
        assert part_psm["frag_stop_idx"].iloc[-1] == len(part_frag)


def test_split_is_reproducible():
    psm, frag = toy_data()
    a = split(psm, frag, 0.34, seed=7)[1][0]["sequence"].tolist()
    b = split(psm, frag, 0.34, seed=7)[1][0]["sequence"].tolist()
    assert a == b


# ---- default library range (length 7-35, charge 2-4) ------------------------------------------

def test_in_default_range_bounds():
    psm = pd.DataFrame({
        "sequence": ["A" * 6, "A" * 7, "A" * 35, "A" * 36, "A" * 10, "A" * 10, "A" * 10, "A" * 10],
        "charge":   [2,        2,        2,         2,         1,        2,        4,        5],
    })
    assert list(in_default_range(psm)) == [False, True, True, False, False, True, True, False]


def test_restrict_to_default_range_keeps_fragments_with_their_spectra(tmp_path):
    from split_heldout import write
    psm = pd.DataFrame({"sequence": ["AAK", "PEPTIDER", "LLLLLLLK", "PEPTIDEK"], "charge": [2, 2, 5, 3]})
    lengths = [len(s) - 1 for s in psm["sequence"]]
    stops = np.cumsum(lengths)
    psm["frag_start_idx"], psm["frag_stop_idx"] = stops - lengths, stops
    frag = pd.DataFrame({"b_z1": [i * 100 + j for i, n in enumerate(lengths) for j in range(n)],
                         "b_z2": 0.0, "y_z1": 0.0, "y_z2": 0.0})
    write(str(tmp_path / "in"), psm, frag)
    kept, total = restrict_to_default_range(str(tmp_path / "in"), str(tmp_path / "out"))
    # AAK is too short and LLLLLLLK is 5+: only the two 8-residue peptides at 2+/3+ remain
    assert (kept, total) == (2, 4)
    out_psm = pd.read_csv(tmp_path / "out" / "psm_pdv.txt", sep="\t")
    out_frag = pd.read_csv(tmp_path / "out" / "fragment_intensity_df.tsv", sep="\t")
    assert list(out_psm["sequence"]) == ["PEPTIDER", "PEPTIDEK"]
    assert list(out_frag["b_z1"][:7]) == [100 + j for j in range(7)]   # PEPTIDER (spectrum 1)
    assert list(out_frag["b_z1"][7:]) == [300 + j for j in range(7)]   # PEPTIDEK (spectrum 3)


# ---- compare_prosit_tmt -----------------------------------------------------------------------

def test_proforma():
    assert proforma("PEPCMK", "TMT6plex@Any N-term;Carbamidomethyl@C;Oxidation@M;TMT6plex@K", "0;4;5;6") \
        == "[UNIMOD:737]-PEPC[UNIMOD:4]M[UNIMOD:35]K[UNIMOD:737]"
    assert proforma("PEPTIDE", "", "") == "PEPTIDE"
    with pytest.raises(ValueError):
        proforma("PEPSK", "Phospho@S", "4")


def test_prosit_fragments_follow_the_heldout_layout():
    # 4 residues -> 3 rows; b_n at row n-1, y_n at row L-n-1; charge 3 and -1 (impossible) dropped
    labels = np.array(["b1+1", "y1+1", "y3+2", "b2+3", "y2+1"], dtype=object)
    values = np.array([0.2, 0.5, 0.7, 0.9, -1.0], dtype=np.float32)
    m = to_fragment_rows(4, values, labels)                  # columns b_z1, b_z2, y_z1, y_z2
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[0, 0], expected[2, 2], expected[0, 3] = 0.2, 0.5, 0.7
    assert np.array_equal(m, expected)


def test_in_prosit_range():
    psm = pd.DataFrame({"sequence": ["A" * 30, "A" * 31, "A" * 10, "A" * 10], "charge": [2, 2, 6, 7]})
    assert list(in_prosit_range(psm)) == [True, False, True, False]


def test_fragment_mz():
    from compare_prosit_tmt import fragment_mz
    # TMT-GAK: b1 = G + TMT + H+, y1 = K + H2O + H+ (row L-2 = 1), b2++ = (G + A + TMT + 2H+)/2
    mz = fragment_mz("GAK", "TMT6plex@Any N-term", "0")
    assert mz[0, 0] == pytest.approx(57.021464 + 229.162932 + 1.007276)
    assert mz[1, 2] == pytest.approx(128.094963 + 18.010565 + 1.007276)
    assert mz[1, 1] == pytest.approx((57.021464 + 71.037114 + 229.162932 + 2 * 1.007276) / 2)
