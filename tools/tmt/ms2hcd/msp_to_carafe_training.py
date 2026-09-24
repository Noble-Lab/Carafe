#!/usr/bin/env python
"""Convert a NIST MSP spectral library into Carafe/AlphaPeptDeep MS2 training data.

Built for the NIST Human CPTAC HCD TMT library (ProteomeXchange PXL000010,
``cptac3_tmt_selected_passed_best.msp``, 597,548 spectra), which already carries annotated
fragment intensities. That means no raw files, no msconvert and no scan matching: this writes
``psm_pdv.txt`` and ``fragment_intensity_df.tsv`` directly, the same two files that
``src/main/resources/py/v2/ai.py`` reads, bypassing Carafe's Java DDA path entirely.

MSP conventions this relies on (verified against the CPTAC3 TMT library, see tools/tmt/ms2hcd/README.md)
--------------------------------------------------------------------------------------------
Name:    ``AAAAAAAAAAAAAAAGAGAGAK/2_2(0,A,TMT6plex)(21,K,TMT6plex)_59eV``
Comment: ``... Mods=2(0,A,TMT6plex)(21,K,TMT6plex) Charge=2 ... FTResolution=35000
           Filter="FTMS + p NSI d Full ms2 1028.0908@hcd32.00 [...]"``
Peaks:   ``585.3561  730822  "b5/0.6ppm,y4-H2O+CO/0.6ppm"``

* Modification positions are **0-based residue indices**: ``(13,C,CAM)`` is the C at 1-based 14.
* Peptide N-terminal modifications are recorded at position 0 carrying the *first residue's*
  letter. Every spectrum in this library has one. Where residue 1 is itself modified you get
  position 0 twice, e.g. ``2(0,K,TMT6plex)(0,K,TMT6plex)`` = N-term label plus the K side chain
  (35,736 such spectra: K 20,323, C 12,218, M 3,195). A TMT label at position 0 is therefore the
  terminus (mod_sites 0) and any other position-0 entry belongs to residue 1 (mod_sites 1). This
  does not depend on the order the entries are listed in, and a library without N-terminal labels
  gets its position-0 residue modifications on residue 1 rather than on the N-terminus.
* Instrument is not stated, but ``FTResolution >= 50000`` means Lumos and lower means Q Exactive.
  That rule agrees with the instrument named in the original file name on 93,777/93,777 spectra.
* NCE comes from the ``@hcd<value>`` field of the Thermo filter string.

Only clean singly and doubly charged b/y ions are kept: no neutral losses, no isotope peaks,
no a ions, no internal or immonium ions, and no TMT reporter ions. Those account for 97.5% of
annotated clean backbone fragments (library_stats.py) and match the ``b_z1/b_z2/y_z1/y_z2`` layout that the
generic AlphaPeptDeep MS2 model predicts.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter

NAME_RE = re.compile(r"^Name:\s*(\S+)")
MODS_RE = re.compile(r"\bMods=(\S+)")
MOD_TOKEN_RE = re.compile(r"\((\d+),([A-Za-z]),([^)]+)\)")
CHARGE_RE = re.compile(r"\bCharge=(\d+)")
NCE_RE = re.compile(r"@hcd([\d.]+)")
FTRES_RE = re.compile(r"\bFTResolution=(\d+)")
NUMPEAKS_RE = re.compile(r"^Num peaks:\s*(\d+)")
# A clean backbone ion: b/y, an ordinal, an optional ^charge. Nothing else.
CLEAN_ION_RE = re.compile(r"^([by])(\d+)(?:\^(\d+))?$")

FRAG_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]

# MSP modification name -> AlphaBase modification name stem.
MOD_NAME_MAP = {
    "TMT6plex": "TMT6plex",
    "TMTpro": "TMTpro",
    "TMT2plex": "TMT2plex",
    "CAM": "Carbamidomethyl",
    "Carbamidomethyl": "Carbamidomethyl",
    "Oxidation": "Oxidation",
}

# Isobaric labels: only these can be the peptide N-terminal modification at MSP position 0.
N_TERMINAL_LABELS = {"TMT6plex", "TMTpro", "TMT2plex"}


def parse_mods(mods_field: str, sequence: str):
    """Translate an MSP ``Mods=`` field into AlphaBase (mods, mod_sites) strings.

    Returns ``(mods, mod_sites)`` or raises ``ValueError`` if anything is unrecognised.
    """
    if not mods_field or mods_field == "0":
        return "", ""

    tokens = MOD_TOKEN_RE.findall(mods_field)
    if not tokens:
        raise ValueError(f"no modification tokens in {mods_field!r}")

    entries = []
    nterm_taken = False
    for pos_str, aa, raw_name in tokens:
        pos = int(pos_str)
        stem = MOD_NAME_MAP.get(raw_name)
        if stem is None:
            raise ValueError(f"unknown modification {raw_name!r}")

        if pos == 0 and stem in N_TERMINAL_LABELS and not nterm_taken:
            # An isobaric label at position 0 is the peptide N-terminal label. Anything else at
            # position 0 (CAM, oxidation, or a second label) is on residue 1, handled below.
            nterm_taken = True
            entries.append((f"{stem}@Any N-term", 0))
            continue

        site = pos + 1 if pos > 0 else 1
        if site > len(sequence):
            raise ValueError(f"modification site {site} beyond peptide length {len(sequence)}")
        residue = sequence[site - 1]
        if residue != aa:
            raise ValueError(f"residue mismatch at {site}: sequence has {residue}, Mods says {aa}")
        entries.append((f"{stem}@{residue}", site))

    entries.sort(key=lambda e: e[1])
    return ";".join(e[0] for e in entries), ";".join(str(e[1]) for e in entries)


def instrument_for(ft_resolution):
    """Lumos above 50k MS2 resolution, Q Exactive below. Validated on 93,777 spectra whose
    original file name states the instrument (100% agreement)."""
    if ft_resolution is None:
        return None
    return "Lumos" if ft_resolution >= 50000 else "QE"


def parse_peak_annotation(annotation: str, first_only: bool):
    """Return ``(ion_type, ordinal, charge)`` for a clean b/y peak, else ``None``."""
    alternatives = annotation.strip('"').split(",")
    if not first_only and len(alternatives) > 1:
        return None
    match = CLEAN_ION_RE.match(alternatives[0].split("/")[0])
    if not match:
        return None
    charge = int(match.group(3) or 1)
    return match.group(1), int(match.group(2)), charge


def iter_spectra(path: str):
    """Stream (name, comment, peaks) records out of an MSP file."""
    name = comment = None
    peaks = []
    in_peaks = False
    with io.open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Name:"):
                if name is not None:
                    yield name, comment, peaks
                name = NAME_RE.match(line).group(1)
                comment = ""
                peaks = []
                in_peaks = False
            elif line.startswith("Comment:"):
                comment = line
            elif line.startswith("Num peaks:"):
                in_peaks = True
            elif in_peaks and line and (line[0].isdigit() or line[0] == "."):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    peaks.append((parts[1], parts[2]))
    if name is not None:
        yield name, comment, peaks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msp", required=True, help="Input .msp library")
    parser.add_argument("--out-dir", required=True, help="Directory to write psm_pdv.txt and fragment_intensity_df.tsv")
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=35)
    parser.add_argument("--min-fragments", type=int, default=6,
                        help="Drop spectra with fewer clean b/y ions than this (default: 6)")
    parser.add_argument("--nce", type=float, default=None,
                        help="Keep only spectra at this NCE (default: keep all)")
    parser.add_argument("--instrument", default=None,
                        help="Override the inferred instrument for every spectrum")
    parser.add_argument("--ambiguous", choices=["first", "drop"], default="first",
                        help="Peaks with several annotations: take the first (NIST orders by "
                             "likelihood, default) or drop them. About 2%% of peaks.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N written spectra")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Write raw intensities instead of scaling each spectrum to max 1")
    args = parser.parse_args(argv)

    try:
        from alphabase.constants.modification import MOD_DF
        known_mods = set(MOD_DF.index)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not load AlphaBase modifications ({exc}); names will not be checked",
              file=sys.stderr)
        known_mods = None

    os.makedirs(args.out_dir, exist_ok=True)
    psm_path = os.path.join(args.out_dir, "psm_pdv.txt")
    frag_path = os.path.join(args.out_dir, "fragment_intensity_df.tsv")

    stats = Counter()
    nce_seen = Counter()
    inst_seen = Counter()
    mods_seen = Counter()
    bad_examples = []
    frag_cursor = 0
    written = 0

    with open(psm_path, "w", newline="", encoding="utf-8") as psm_handle, \
            open(frag_path, "w", newline="", encoding="utf-8") as frag_handle:

        psm_writer = csv.writer(psm_handle, delimiter="\t", lineterminator="\n")
        psm_writer.writerow([
            "sequence", "charge", "mods", "mod_sites", "nce", "instrument",
            "frag_start_idx", "frag_stop_idx", "n_frags",
        ])
        frag_handle.write("\t".join(FRAG_COLUMNS) + "\n")

        for name, comment, peaks in iter_spectra(args.msp):
            stats["read"] += 1
            sequence = name.split("/")[0]

            if not args.min_length <= len(sequence) <= args.max_length:
                stats["drop:length"] += 1
                continue
            if not sequence.isalpha() or not sequence.isupper():
                stats["drop:sequence"] += 1
                continue

            charge_match = CHARGE_RE.search(comment)
            if not charge_match:
                stats["drop:no_charge"] += 1
                continue
            charge = int(charge_match.group(1))
            if not args.min_charge <= charge <= args.max_charge:
                stats["drop:charge"] += 1
                continue

            nce_match = NCE_RE.search(comment)
            if not nce_match:
                stats["drop:no_nce"] += 1
                continue
            nce = float(nce_match.group(1))
            if args.nce is not None and abs(nce - args.nce) > 1e-6:
                stats["drop:nce_filter"] += 1
                continue

            if args.instrument:
                instrument = args.instrument
            else:
                ftres = FTRES_RE.search(comment)
                instrument = instrument_for(int(ftres.group(1)) if ftres else None)
                if instrument is None:
                    stats["drop:no_instrument"] += 1
                    continue

            mods_match = MODS_RE.search(comment)
            try:
                mods, mod_sites = parse_mods(mods_match.group(1) if mods_match else "0", sequence)
            except ValueError as exc:
                stats["drop:bad_mods"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{name}: {exc}")
                continue

            if known_mods is not None:
                unknown = [m for m in mods.split(";") if m and m not in known_mods]
                if unknown:
                    stats["drop:unknown_mod_name"] += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"{name}: not in AlphaBase: {unknown}")
                    continue

            # Build the (len-1) x 4 fragment matrix.
            n_rows = len(sequence) - 1
            matrix = [[0.0] * len(FRAG_COLUMNS) for _ in range(n_rows)]
            for intensity_str, annotation in peaks:
                parsed = parse_peak_annotation(annotation, args.ambiguous == "first")
                if parsed is None:
                    continue
                ion_type, ordinal, frag_charge = parsed
                if frag_charge > 2 or ordinal < 1 or ordinal > n_rows:
                    continue
                # AlphaBase row order: b_n at row n-1, y_n at row L-n-1.
                row = ordinal - 1 if ion_type == "b" else n_rows - ordinal
                if not 0 <= row < n_rows:
                    continue
                column = FRAG_COLUMNS.index(f"{ion_type}_z{frag_charge}")
                try:
                    intensity = float(intensity_str)
                except ValueError:
                    continue
                if intensity > matrix[row][column]:
                    matrix[row][column] = intensity

            # Count distinct populated cells, not peaks: several peaks can map to one cell.
            n_frags = sum(1 for row in matrix for value in row if value > 0)

            if n_frags < args.min_fragments:
                stats["drop:too_few_fragments"] += 1
                continue

            if not args.no_normalize:
                peak_max = max(max(row) for row in matrix)
                if peak_max > 0:
                    matrix = [[v / peak_max for v in row] for row in matrix]

            for row in matrix:
                frag_handle.write("\t".join(f"{v:.6g}" for v in row) + "\n")

            psm_writer.writerow([
                sequence, charge, mods, mod_sites, f"{nce:g}", instrument,
                frag_cursor, frag_cursor + n_rows, n_frags,
            ])
            frag_cursor += n_rows
            written += 1
            stats["written"] += 1
            nce_seen[f"{nce:g}"] += 1
            inst_seen[instrument] += 1
            for m in mods.split(";"):
                if m:
                    mods_seen[m] += 1

            if written % 100000 == 0:
                print(f"  ...{written} spectra written", file=sys.stderr)
            if args.limit and written >= args.limit:
                break

    meta = {
        "source_msp": os.path.basename(args.msp),  # file name only: no local paths in shared output
        "spectra_read": stats["read"],
        "spectra": written,
        "dropped": {key.split(":", 1)[1]: stats[key] for key in sorted(stats) if key.startswith("drop:")},
        "fragment_rows": frag_cursor,
        "nce": dict(nce_seen.most_common()),
        "instrument": dict(inst_seen.most_common()),
        "modifications": dict(mods_seen.most_common()),
    }
    with open(os.path.join(args.out_dir, "msp_conversion.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    for key in sorted(stats):
        print(f"{key}\t{stats[key]}", file=sys.stderr)
    if bad_examples:
        print("\nexamples of rejected records:", file=sys.stderr)
        for example in bad_examples:
            print(f"  {example}", file=sys.stderr)
    print(f"\nNCE:         {dict(nce_seen.most_common())}", file=sys.stderr)
    print(f"instrument:  {dict(inst_seen.most_common())}", file=sys.stderr)
    print(f"modifications: {dict(mods_seen.most_common())}", file=sys.stderr)
    print(f"\nwrote {written} spectra and {frag_cursor} fragment rows to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
