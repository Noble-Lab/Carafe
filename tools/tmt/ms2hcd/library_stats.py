#!/usr/bin/env python
"""Describe the NIST CPTAC3 TMT spectral library: every number README.md quotes about it.

One streaming pass over ``cptac3_tmt_selected_passed_best.msp`` (ProteomeXchange PXL000010, 7.3 GB).
Reports, for the library as distributed (before any filtering by msp_to_carafe_training.py):

* spectra, unique peptide sequences and unique modified peptides
* modification counts, precursor charge, NCE and peptide length distributions
* the N-terminal convention: every spectrum carries a modification at position 0, and how many
  carry two (N-terminal label plus a label on residue 1)
* the instrument rule used for training (FTResolution >= 50000 -> Lumos, else Q Exactive),
  checked against spectra whose original file name states the instrument
* fragment yield: clean b/y ions (no neutral loss or isotope, first annotation alternative) per
  spectrum, the share of spectra with at least 6 and 10, ambiguous peaks, and the share of clean
  backbone ions at charge <= 2 (what the b_z1/b_z2/y_z1/y_z2 training layout can represent)

    python library_stats.py --msp cptac3_tmt_selected_passed_best.msp --out library_stats.json

The MSP is sorted alphabetically, so a partial read is not representative; ``--stride N`` takes
every Nth spectrum for a quick look, the default reads everything.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
from collections import Counter

MODS_RE = re.compile(r"\bMods=(\S+)")
MOD_TOKEN_RE = re.compile(r"\((\d+),([A-Za-z]),([^)]+)\)")
CHARGE_RE = re.compile(r"\bCharge=(\d+)")
NCE_RE = re.compile(r"@hcd([\d.]+)")
FTRES_RE = re.compile(r"\bFTResolution=(\d+)")
ORIGFILE_RE = re.compile(r'Origfile="([^"]+)"')
CLEAN_ION_RE = re.compile(r"^([by])(\d+)(?:\^(\d+))?$")
LUMOS_TOKEN = re.compile(r"(^|[_\-])LUMOS([_\-.]|$)")
QE_TOKEN = re.compile(r"(^|[_\-])QE([_\-.]|$)")

LUMOS_MIN_RESOLUTION = 50000  # the rule msp_to_carafe_training.py uses


def instrument_from_resolution(resolution: int) -> str:
    return "Lumos" if resolution >= LUMOS_MIN_RESOLUTION else "QE"


def instrument_from_filename(origfile: str):
    """The instrument an original file name states, or None when it names none."""
    name = origfile.upper()
    if LUMOS_TOKEN.search(name):
        return "Lumos"
    if QE_TOKEN.search(name):
        return "QE"
    return None


class Spectrum:
    __slots__ = ("sequence", "name", "length", "clean", "peaks")

    def __init__(self, name: str):
        self.name = name
        self.sequence = name.split("/")[0]
        self.length = len(self.sequence)
        self.clean = 0
        self.peaks = 0


def collect(msp_path: str, stride: int = 1) -> dict:
    n_total = 0
    sequences, modified = set(), set()
    mods, charges, nces, lengths, instruments = Counter(), Counter(), Counter(), Counter(), Counter()
    no_nterm = two_nterm = 0
    two_nterm_by_residue1 = Counter()  # what the second position-0 entry is, e.g. 'K TMT6plex'
    label_not_first = 0  # msp_to_carafe_training.py takes the FIRST position-0 entry as the N-terminus
    rule_agree, rule_checked = Counter(), 0
    clean_per_spectrum, matrix_fraction = [], []
    ambiguous = annotated = 0
    backbone_by_charge = Counter()

    current = None
    take = False
    with io.open(msp_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("Name: "):
                if current is not None:
                    clean_per_spectrum.append(current.clean)
                    matrix_fraction.append(current.clean / max(1, 4 * (current.length - 1)))
                take = n_total % stride == 0
                n_total += 1
                current = Spectrum(line[6:].strip()) if take else None
                if take:
                    sequences.add(current.sequence)
                    # Name: SEQUENCE/charge_mods_eV, e.g. AAK/2_2(0,A,TMT6plex)(2,K,TMT6plex)_59eV
                    rest = current.name.split("/", 1)[1] if "/" in current.name else ""
                    mods_part = rest.split("_")[1] if rest.count("_") >= 1 else ""
                    modified.add(current.sequence + "|" + mods_part)
                    lengths[current.length] += 1
            elif not take:
                continue
            elif line.startswith("Comment: "):
                m = CHARGE_RE.search(line)
                if m:
                    charges[int(m.group(1))] += 1
                m = NCE_RE.search(line)
                nces[m.group(1) if m else "none"] += 1
                m = MODS_RE.search(line)
                tokens = MOD_TOKEN_RE.findall(m.group(1)) if m else []
                for _pos, _aa, name in tokens:
                    mods[name] += 1
                at_zero = sum(1 for pos, _aa, _name in tokens if pos == "0")
                if at_zero == 0:
                    no_nterm += 1
                elif at_zero >= 2:
                    two_nterm += 1
                    extra = [f"{aa} {name}" for pos, aa, name in tokens if pos == "0"][1:]
                    two_nterm_by_residue1.update(extra)
                    first = [name for pos, _aa, name in tokens if pos == "0"][0]
                    if not first.startswith("TMT"):
                        label_not_first += 1
                r = FTRES_RE.search(line)
                if r:
                    predicted = instrument_from_resolution(int(r.group(1)))
                    instruments[predicted] += 1
                    o = ORIGFILE_RE.search(line)
                    stated = instrument_from_filename(o.group(1)) if o else None
                    if stated is not None:
                        rule_checked += 1
                        rule_agree[stated == predicted] += 1
                else:
                    instruments["no FTResolution"] += 1
            elif line[:1].isdigit():
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                current.peaks += 1
                annotation = parts[2].strip('"')
                alternatives = annotation.split(",")
                annotated += 1
                if len(alternatives) > 1:
                    ambiguous += 1
                ion = CLEAN_ION_RE.match(alternatives[0].split("/")[0])
                if ion:
                    z = int(ion.group(3) or 1)
                    backbone_by_charge[z] += 1
                    if z <= 2:
                        current.clean += 1
    if current is not None:
        clean_per_spectrum.append(current.clean)
        matrix_fraction.append(current.clean / max(1, 4 * (current.length - 1)))

    n = len(clean_per_spectrum)
    backbone = sum(backbone_by_charge.values())
    return {
        "source_msp": os.path.basename(msp_path),  # file name only: no local paths in shared output
        "stride": stride,
        "spectra_in_file": n_total,
        "spectra_analysed": n,
        "unique_sequences": len(sequences),
        "unique_modified_peptides": len(modified),
        "modifications": dict(mods.most_common()),
        "charge": {str(k): v for k, v in sorted(charges.items())},
        "nce": dict(sorted(nces.items(), key=lambda kv: -kv[1])),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "spectra_without_position0_modification": no_nterm,
        "spectra_with_two_position0_modifications": two_nterm,
        "second_position0_modification": dict(two_nterm_by_residue1.most_common()),
        "two_position0_where_label_is_not_first": label_not_first,
        "instrument_by_resolution_rule": dict(instruments.most_common()),
        "instrument_rule_checked_against_filename": rule_checked,
        "instrument_rule_agrees": rule_agree[True],
        "clean_by_per_spectrum_median": statistics.median(clean_per_spectrum),
        "clean_by_fraction_of_matrix_median": statistics.median(matrix_fraction),
        "spectra_with_at_least_6_clean_by": sum(1 for c in clean_per_spectrum if c >= 6) / n,
        "spectra_with_at_least_10_clean_by": sum(1 for c in clean_per_spectrum if c >= 10) / n,
        "ambiguous_peak_fraction": ambiguous / max(1, annotated),
        "clean_backbone_ions_by_charge": {str(k): v for k, v in sorted(backbone_by_charge.items())},
        "clean_backbone_ions_at_charge_le2": (backbone_by_charge[1] + backbone_by_charge[2]) / max(1, backbone),
    }


def report(s: dict) -> str:
    charge = s["charge"]
    high = sum(v for k, v in charge.items() if int(k) >= 5)
    lines = [
        f"spectra                      {s['spectra_analysed']:,} analysed of {s['spectra_in_file']:,} (stride {s['stride']})",
        f"unique sequences             {s['unique_sequences']:,}",
        f"unique modified peptides     {s['unique_modified_peptides']:,}",
        "modifications                " + "; ".join(f"{k} {v:,}" for k, v in s["modifications"].items()),
        "charge                       " + "; ".join(f"{k}: {charge[k]:,}" for k in ("1", "2", "3", "4") if k in charge)
        + f"; 5+: {high:,}",
        "NCE                          " + "; ".join(f"{k}: {v:,}" for k, v in s["nce"].items()),
        f"length                       {s['length_min']}-{s['length_max']}",
        f"no position-0 modification   {s['spectra_without_position0_modification']:,} spectra",
        f"two position-0 modifications {s['spectra_with_two_position0_modifications']:,} spectra (N-term label + modified residue 1: "
        + "; ".join(f"{k} {v:,}" for k, v in s["second_position0_modification"].items()) + ")",
        f"  ...with the label not first {s['two_position0_where_label_is_not_first']:,} (converter takes the first as N-term)",
        "instrument (resolution rule) " + "; ".join(f"{k} {v:,}" for k, v in s["instrument_by_resolution_rule"].items()),
        f"rule vs instrument in name   agrees on {s['instrument_rule_agrees']:,}/{s['instrument_rule_checked_against_filename']:,}",
        f"clean b/y (z<=2) / spectrum  median {s['clean_by_per_spectrum_median']:g}"
        f" ({s['clean_by_fraction_of_matrix_median']:.0%} of the 4x(L-1) matrix)",
        f"spectra with >=6 / >=10      {s['spectra_with_at_least_6_clean_by']:.1%} / {s['spectra_with_at_least_10_clean_by']:.1%}",
        f"ambiguous peaks              {s['ambiguous_peak_fraction']:.1%}",
        f"clean backbone ions at z<=2  {s['clean_backbone_ions_at_charge_le2']:.1%}",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msp", required=True)
    ap.add_argument("--out", help="write the numbers as JSON")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth spectrum (default: all)")
    args = ap.parse_args(argv)
    stats = collect(args.msp, args.stride)
    print(report(stats))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
