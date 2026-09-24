#!/usr/bin/env python
"""Compare Carafe MS2 models with Prosit_2020_intensity_TMT (served by Koina) on the held-out test set.

Prosit takes peptides of up to 30 residues at precursor charge 1-6, so every model is scored on the
held-out spectra inside that range; the rest are counted, not scored. All models are scored on the
same four columns (b and y ions at fragment charge 1 and 2, the ions the held-out table holds), with
Carafe's calc_ms2_similarity_mask. Prosit's fragment-charge-3 outputs are dropped. SA and COS match
evaluate_heldout.py; its PCC and SPC also count the (all-zero) neutral-loss columns, so they differ
slightly from the values here.

Prosit is scored twice: at each spectrum's recorded NCE, and at a calibrated NCE (recorded NCE plus
one offset per instrument, -6 to +6). The offsets are chosen on a sample of training spectra (median
SA), never on the held-out set. Carafe models use the recorded NCE, the NCE they were trained on.

Writes to --out-dir:
    calibration.json               median Prosit SA per offset and instrument, and the chosen offsets
                                   (reused on later runs made from the same inputs)
    prosit_pred_<offsets>.tsv      Prosit predictions for the held-out set (reused on later runs for
                                   the held-out set recorded in prosit_pred_heldout.json)
    comparison.json                the scores
    mirror_spectra.json            every peak of the plotted spectra, read from --msp
    mirror_light.png, _dark.png    every peak of a few held-out spectra (top) and the predictions
                                   (bottom), picked at random with a fixed seed, one per precursor
                                   charge 2-5; the last --model beside calibrated Prosit

    python compare_prosit_tmt.py --heldout split/heldout --train split/train --msp <library>.msp \
        --model generic --model D=sweep/D/ms2_model.pt --out-dir prosit
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_heldout import MODELS_PY_DIR, load_heldout  # noqa: E402
from split_heldout import in_default_range, reindex  # noqa: E402

KOINA_URL = "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_TMT/infer"
MAX_LENGTH, MAX_CHARGE = 30, 6
UNIMOD = {"TMT6plex": 737, "Carbamidomethyl": 4, "Oxidation": 35}
FRAG_COLUMNS = ["b_z1", "b_z2", "y_z1", "y_z2"]
OFFSETS = range(-6, 7)
SEED = 2024


# ---- Prosit via Koina -------------------------------------------------------------------------

def in_prosit_range(psm: pd.DataFrame) -> np.ndarray:
    return ((psm["sequence"].str.len() <= MAX_LENGTH) & (psm["charge"] >= 1)
            & (psm["charge"] <= MAX_CHARGE)).to_numpy()


def proforma(sequence: str, mods: str, sites: str) -> str:
    """Carafe mods/mod_sites (site 0 = N-terminus, 1-based residues) as a ProForma string."""
    residues, nterm = list(sequence), ""
    for mod, site in zip(mods.split(";"), sites.split(";")) if mods else ():
        name = mod.split("@")[0]
        if name not in UNIMOD:
            raise ValueError(f"no UNIMOD id for modification {mod!r} in {sequence}")
        tag = f"[UNIMOD:{UNIMOD[name]}]"
        if int(site) == 0:
            nterm = tag + "-"
        else:
            residues[int(site) - 1] += tag
    return nterm + "".join(residues)


def koina_predict(peptides, charges, nces, url=KOINA_URL, attempts=3):
    """Prosit intensities and annotations (each n x 174) for one request."""
    n = len(peptides)
    body = json.dumps({"id": "0", "inputs": [
        {"name": "peptide_sequences", "shape": [n, 1], "datatype": "BYTES", "data": list(peptides)},
        {"name": "precursor_charges", "shape": [n, 1], "datatype": "INT32", "data": [int(c) for c in charges]},
        {"name": "collision_energies", "shape": [n, 1], "datatype": "FP32", "data": [float(e) for e in nces]},
        {"name": "fragmentation_types", "shape": [n, 1], "datatype": "BYTES", "data": ["HCD"] * n},
    ], "outputs": [{"name": "intensities"}, {"name": "annotation"}]}).encode()
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                outputs = {o["name"]: o for o in json.load(response)["outputs"]}
            break
        except urllib.error.HTTPError as error:
            reason = error.read().decode("utf-8", "replace")[:500]
            if error.code < 500:  # the server rejected the request itself; asking again cannot help
                raise SystemExit(f"Koina rejected the request: {error} {reason}")
            if attempt == attempts:
                raise SystemExit(f"Koina request failed {attempts} times: {error} {reason}")
            print(f"Koina request failed ({error}); retrying", flush=True)
        # socket.timeout is not a TimeoutError before Python 3.10
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            if attempt == attempts:
                raise SystemExit(f"Koina request failed {attempts} times: {error}")
            print(f"Koina request failed ({error}); retrying", flush=True)
        time.sleep(10 * attempt)
    width = outputs["intensities"]["shape"][1]
    intensities = np.asarray(outputs["intensities"]["data"], dtype=np.float32).reshape(n, width)
    annotation = np.asarray(outputs["annotation"]["data"], dtype=object).reshape(n, width)
    return intensities, annotation


def to_fragment_rows(length: int, intensities, annotation) -> np.ndarray:
    """One Prosit spectrum in the held-out layout: b_n at row n-1, y_n at row L-n-1."""
    n_rows = length - 1
    matrix = np.zeros((n_rows, len(FRAG_COLUMNS)), dtype=np.float32)
    for value, label in zip(intensities, annotation):  # label: "y12+2"
        ordinal, charge = (int(x) for x in label[1:].split("+"))
        if value <= 0 or charge > 2 or ordinal > n_rows:  # -1 marks ions Prosit rules out
            continue
        row = ordinal - 1 if label[0] == "b" else n_rows - ordinal
        matrix[row, FRAG_COLUMNS.index(f"{label[0]}_z{charge}")] = value
    return matrix


def prosit_predict(requests, n_fragment_rows, batch_size, workers):
    """Prosit predictions for several (psm, nces) sets at once, as fragment tables in their layouts.

    Batches from all sets go to Koina together, `workers` requests at a time."""
    jobs = []
    for k, (psm, nces) in enumerate(requests):
        peptides = [proforma(s, m, t) for s, m, t in zip(psm["sequence"], psm["mods"], psm["mod_sites"])]
        for i in range(0, len(psm), batch_size):
            jobs.append((k, i, peptides[i:i + batch_size], psm["charge"].iloc[i:i + batch_size],
                         np.asarray(nces)[i:i + batch_size]))
    tables = [np.zeros((n, len(FRAG_COLUMNS)), dtype=np.float32) for n in n_fragment_rows]
    start, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (k, i, *_), (intensities, annotation) in zip(jobs, pool.map(lambda j: koina_predict(*j[2:]), jobs)):
            batch = requests[k][0].iloc[i:i + batch_size]
            for row, (length, a, b) in enumerate(zip(batch["sequence"].str.len(), batch["frag_start_idx"],
                                                     batch["frag_stop_idx"])):
                tables[k][a:b] = to_fragment_rows(length, intensities[row], annotation[row])
            done += 1
            print(f"Koina: {done} of {len(jobs)} requests ({time.time() - start:.0f} s)", flush=True)
    return [pd.DataFrame(t, columns=FRAG_COLUMNS) for t in tables]


# ---- scoring ----------------------------------------------------------------------------------

def similarity(psm, predicted, observed, gpu):
    sys.path.insert(0, MODELS_PY_DIR)
    from models import calc_ms2_similarity_mask
    result, _ = calc_ms2_similarity_mask(psm.copy(), predicted[FRAG_COLUMNS], observed[FRAG_COLUMNS], None, GPU=gpu)
    return result[result.index.isin(psm.index)]  # drop the four appended summary rows


def carafe_predict(spec, psm, observed, device):
    """Carafe/AlphaPeptDeep MS2 prediction ('generic', a .pt path, or name=path), in psm's layout."""
    sys.path.insert(0, MODELS_PY_DIR)
    import torch
    from models import ModelManager
    name, _, path = spec.partition("=") if "=" in spec else (spec, "", spec)
    manager = ModelManager(mask_modloss=True, device=device)
    manager.load_installed_models("generic", model_list=["ms2"])
    if path != "generic":
        # load_external_models only warns on failure, which would score the generic model instead
        if not os.path.isfile(path):
            raise SystemExit(f"{name}: model file not found: {path}")
        before = [p.detach().clone() for p in manager.ms2_model.model.parameters()]
        manager.load_external_models(ms2_model_file=path)
        if all(torch.equal(b, a.detach()) for b, a in zip(before, manager.ms2_model.model.parameters())):
            raise SystemExit(f"{name}: {path} did not load (the weights are still the generic model's)")
    reference = pd.DataFrame(0.0, index=observed.index, columns=manager.ms2_model.charged_frag_types)
    return name, manager.ms2_model.predict(psm.copy(), reference_frag_df=reference)


def summarize(name, result):
    def medians(part):
        return {"spectra": int(len(part)),
                "PCC": float(part["PCC"].median()), "COS": float(part["COS"].median()),
                "SA": float(part["SA"].median()), "SPC": float(part["SPC"].median()),
                "SA_gt_0.9": float((part["SA"] > 0.9).mean())}
    length_bin = result["sequence"].str.len().map(lambda n: "<7" if n < 7 else (">35" if n > 35 else "7-35"))
    return {"model": name, **medians(result), "default_range": medians(result[in_default_range(result)]),
            "SA_by_instrument": {str(k): float(v) for k, v in result.groupby("instrument")["SA"].median().items()},
            "SA_by_charge": {str(k): [float(v), int(n)] for k, (v, n) in
                             result.groupby(result["charge"].astype(int))["SA"].agg(["median", "size"]).iterrows()},
            "SA_by_length": {str(k): [float(v), int(n)] for k, (v, n) in
                             result.groupby(length_bin)["SA"].agg(["median", "size"]).iterrows()}}


# ---- NCE calibration on training spectra ------------------------------------------------------

def data_signature(directory):
    """Byte sizes of a data directory's two files: enough to tell one split from another."""
    return {name: os.path.getsize(os.path.join(directory, name))
            for name in ("psm_pdv.txt", "fragment_intensity_df.tsv")}


def calibrate(train_dir, per_instrument, batch_size, workers, gpu, out_path):
    """Per-instrument NCE offset for Prosit with the highest median SA on sampled training spectra.

    Reused from out_path when it was made from the same training data, sample size, seed and offsets."""
    inputs = {"train_data": data_signature(train_dir), "spectra_per_instrument_requested": per_instrument,
              "seed": SEED, "offsets": list(OFFSETS)}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            saved = json.load(fh)
        if saved.get("inputs") == inputs:
            print(f"read NCE calibration from {out_path}", flush=True)
            return {k: int(v) for k, v in saved["chosen_offsets"].items()}
        print(f"{out_path} was made from other inputs; calibrating again", flush=True)
    psm, observed = load_heldout(train_dir)
    psm = psm[in_prosit_range(psm)]
    sample = pd.concat([g.sample(min(per_instrument, len(g)), random_state=SEED)
                        for _, g in psm.groupby("instrument")])
    sample, sample_obs = reindex(sample, observed)
    predictions = prosit_predict([(sample, sample["nce"] + o) for o in OFFSETS], [len(sample_obs)] * len(OFFSETS),
                                 batch_size, workers)
    table = {}
    for offset, predicted in zip(OFFSETS, predictions):
        result = similarity(sample, predicted, sample_obs, gpu)
        table[offset] = {str(k): float(v) for k, v in result.groupby("instrument")["SA"].median().items()}
    instruments = sorted(sample["instrument"].unique())
    chosen = {i: max(OFFSETS, key=lambda o: table[o][i]) for i in instruments}
    for offset in OFFSETS:
        print(f"offset {offset:+d}: " + "  ".join(f"{i} {table[offset][i]:.4f}" for i in instruments), flush=True)
    print("chosen offsets: " + ", ".join(f"{i} {o:+d}" for i, o in chosen.items()), flush=True)
    with open(out_path, "w") as fh:
        json.dump({"inputs": inputs,
                   "spectra_per_instrument": {i: int((sample["instrument"] == i).sum()) for i in instruments},
                   "median_SA": {f"{o:+d}": table[o] for o in OFFSETS},
                   "chosen_offsets": chosen}, fh, indent=2)
    return chosen


# ---- mirror plots -----------------------------------------------------------------------------

RESIDUE_MASS = {
    "G": 57.021464, "A": 71.037114, "S": 87.032028, "P": 97.052764, "V": 99.068414, "T": 101.047679,
    "C": 103.009185, "L": 113.084064, "I": 113.084064, "N": 114.042927, "D": 115.026943, "Q": 128.058578,
    "K": 128.094963, "E": 129.042593, "M": 131.040485, "H": 137.058912, "F": 147.068414, "R": 156.101111,
    "Y": 163.063329, "W": 186.079313,
}
MOD_MASS = {"TMT6plex": 229.162932, "Carbamidomethyl": 57.021464, "Oxidation": 15.994915}
PROTON, WATER = 1.007276, 18.010565


def fragment_mz(sequence, mods, sites):
    """m/z of b/y ions at charge 1-2 in the held-out layout (rows x FRAG_COLUMNS)."""
    masses = [RESIDUE_MASS[aa] for aa in sequence]
    for mod, site in zip(mods.split(";"), sites.split(";")) if mods else ():
        masses[max(int(site), 1) - 1] += MOD_MASS[mod.split("@")[0]]  # site 0 (N-term) sits on residue 1
    prefix = np.cumsum(masses)[:-1]              # b_n neutral residue sum, n = 1..L-1 (row n-1)
    suffix = sum(masses) - prefix + WATER         # y_(L-n) at the same row
    return np.column_stack([(prefix + PROTON), (prefix + 2 * PROTON) / 2, (suffix + PROTON), (suffix + 2 * PROTON) / 2])


def pick_spectra(psm):
    """One held-out spectrum per precursor charge 2-5 inside Prosit's range, at random (fixed seed)."""
    candidates = psm[in_prosit_range(psm)]
    return [candidates[candidates["charge"] == z].sample(1, random_state=SEED).index[0]
            for z in (2, 3, 4, 5) if (candidates["charge"] == z).any()]


def spectrum_key(sequence, charge, mods, sites, nce, instrument):
    return f"{sequence}/{int(charge)}/{mods}/{sites}/{float(nce):g}/{instrument}"


def original_spectra(msp, psm, observed, picks, cache_path):
    """Every peak of each picked held-out spectrum, from the NIST MSP it was converted from.

    A spectrum is found by sequence, charge, modifications, NCE and instrument; if several MSP entries
    share these, the one whose clean b/y ions equal the held-out row is taken. Returns
    {index: (mz, intensity, label)}, label naming the peaks that became the spectrum's b/y ions (the
    strongest peak per ion at fragment charge 1-2, as msp_to_carafe_training.py keeps them) and ""
    for every other peak. Results are cached in cache_path."""
    import msp_to_carafe_training as conv

    keys = {i: spectrum_key(*psm.loc[i, ["sequence", "charge", "mods", "mod_sites", "nce", "instrument"]])
            for i in picks}
    found = {}
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            found = json.load(fh)
    todo = {k: i for i, k in keys.items() if k not in found}
    if todo:
        sequences = {k.split("/")[0] for k in todo}
        start = time.time()
        for name, comment, peaks in read_msp(msp, sequences):
            sequence = name.split("/")[0]
            charge, nce = conv.CHARGE_RE.search(comment), conv.NCE_RE.search(comment)
            ftres, mods_field = conv.FTRES_RE.search(comment), conv.MODS_RE.search(comment)
            if not (charge and nce and ftres):
                continue
            try:
                mods, sites = conv.parse_mods(mods_field.group(1) if mods_field else "0", sequence)
            except ValueError:
                continue
            key = spectrum_key(sequence, charge.group(1), mods, sites, nce.group(1),
                               conv.instrument_for(int(ftres.group(1))))
            if key not in todo or key in found:
                continue
            n_rows = len(sequence) - 1
            matrix, source = np.zeros((n_rows, len(FRAG_COLUMNS))), {}
            for k, (_, intensity, annotation) in enumerate(peaks):
                parsed = conv.parse_peak_annotation(annotation, True)
                if parsed is None or parsed[2] > 2 or not 1 <= parsed[1] <= n_rows:
                    continue
                ion, ordinal, z = parsed
                cell = (ordinal - 1 if ion == "b" else n_rows - ordinal, FRAG_COLUMNS.index(f"{ion}_z{z}"))
                if intensity > matrix[cell]:
                    matrix[cell], source[cell] = intensity, k
            i = todo[key]
            expected = observed.iloc[psm.loc[i, "frag_start_idx"]:psm.loc[i, "frag_stop_idx"]][FRAG_COLUMNS].to_numpy()
            if matrix.max() <= 0 or not np.allclose(matrix / matrix.max(), expected, rtol=1e-4, atol=1e-6):
                continue
            labels = [""] * len(peaks)
            for (row, col), k in source.items():
                ion, z = FRAG_COLUMNS[col][0], int(FRAG_COLUMNS[col][-1])
                labels[k] = f"{ion}{row + 1 if ion == 'b' else n_rows - row}" + ("++" if z == 2 else "")
            found[key] = {"mz": [q[0] for q in peaks], "intensity": [q[1] for q in peaks], "label": labels}
            if all(k in found for k in todo):
                break
        missing = [k for k in todo if k not in found]
        if missing:
            raise SystemExit(f"not found in {msp}: {missing}")
        print(f"read {len(todo)} spectra from {msp} ({time.time() - start:.0f} s)", flush=True)
        with open(cache_path, "w") as fh:
            json.dump(found, fh)
    return {i: (np.array(found[k]["mz"]), np.array(found[k]["intensity"]), found[k]["label"]) for i, k in keys.items()}


def read_msp(path, sequences):
    """(name, comment, [(mz, intensity, annotation)]) for MSP entries whose sequence is in `sequences`."""
    import msp_to_carafe_training as conv
    name, comment, peaks, keep = None, "", [], False
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("Name:"):
                if keep:
                    yield name, comment, peaks
                name = conv.NAME_RE.match(line).group(1)
                keep, comment, peaks = name.split("/")[0] in sequences, "", []
            elif not keep:
                continue
            elif line.startswith("Comment:"):
                comment = line
            elif line[:1].isdigit() or line[:1] == ".":
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    peaks.append((float(parts[0]), float(parts[1]), parts[2]))
    if keep:
        yield name, comment, peaks


def mirror_plots(psm, originals, predictions, scores, picks, out_prefix):
    """originals: {index: (mz, intensity, label)} from original_spectra; predictions/scores:
    {panel title: fragment table / per-spectrum result} in psm's layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    from plot_loss_curve import THEMES

    for mode, theme in THEMES.items():
        colours = {"b": theme["train"], "y": theme["val"], "": theme["muted"]}
        fig, axes = plt.subplots(len(picks), len(predictions), figsize=(5.2 * len(predictions), 2.6 * len(picks)),
                                 dpi=100, squeeze=False, gridspec_kw={"hspace": 0.75, "wspace": 0.12})
        fig.patch.set_facecolor(theme["surface"])
        for r, index in enumerate(picks):
            p = psm.loc[index]
            a, b = p["frag_start_idx"], p["frag_stop_idx"]
            mz = fragment_mz(p["sequence"], p["mods"], p["mod_sites"])
            peak_mz, peak_intensity, labels = originals[index]
            used = np.array([bool(label) for label in labels])
            relative = peak_intensity / peak_intensity.max()
            # predictions are drawn with their strongest ion as high as the strongest observed b/y ion
            top_ion = relative[used].max()
            # one m/z range per row: every observed peak and predicted ions of at least 1% of the maximum
            ends = [peak_mz.min(), peak_mz.max()]
            for predicted in predictions.values():
                pred = predicted.iloc[a:b][FRAG_COLUMNS].to_numpy()
                ends += [mz[pred >= 0.01 * pred.max()].min(), mz[pred >= 0.01 * pred.max()].max()]
            xlim = (min(ends) - 40, max(ends) + 40)
            for c, (title, predicted) in enumerate(predictions.items()):
                ax = axes[r][c]
                pred = predicted.iloc[a:b][FRAG_COLUMNS].to_numpy()
                pred = pred / pred.max() * top_ion if pred.max() > 0 else pred
                ax.vlines(peak_mz[~used], 0, relative[~used], color=colours[""], linewidth=0.8)
                for ion in ("b", "y"):
                    mine = np.array([label.startswith(ion) for label in labels])
                    ax.vlines(peak_mz[mine], 0, relative[mine], color=colours[ion], linewidth=1.2)
                for col, column in enumerate(FRAG_COLUMNS):
                    ax.vlines(mz[:, col], 0, -pred[:, col], color=colours[column[0]], linewidth=1.2)
                # label the three strongest observed b/y ions
                for k in [k for k in np.argsort(peak_intensity)[::-1] if used[k]][:3]:
                    ax.annotate(labels[k], (peak_mz[k], relative[k]), xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=6.5, color=theme["ink2"])
                ax.axhline(0, color=theme["axis"], linewidth=0.8)
                ax.set_ylim(-1.15, 1.15)
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}"))
                ax.set_facecolor(theme["surface"])
                for side in ("top", "right", "left"):
                    ax.spines[side].set_visible(False)
                ax.spines["bottom"].set_color(theme["axis"])
                ax.tick_params(colors=theme["muted"], labelsize=7, length=0)
                ax.grid(axis="y", color=theme["grid"], linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_xlim(*xlim)
                ax.set_title(f"{title}   SA {scores[title].loc[index, 'SA']:.3f}", fontsize=8,
                             color=theme["ink2"], loc="left", pad=4)
                if c == 0:
                    # the spectrum, as a row header above both panels
                    ax.annotate(f"{proforma(p['sequence'], p['mods'], p['mod_sites'])}   {int(p['charge'])}+   "
                                f"NCE {p['nce']:g}   {p['instrument']}   {len(peak_mz)} peaks", (0, 1.22),
                                xycoords="axes fraction", fontsize=8.5, color=theme["ink"], annotation_clip=False)
                    ax.set_ylabel("observed  |  predicted", fontsize=7, color=theme["ink2"])
                if r == len(picks) - 1:
                    ax.set_xlabel("m/z", fontsize=7, color=theme["ink2"])
        handles = [plt.Line2D([], [], color=colours[k], linewidth=2, label=text)
                   for k, text in (("b", "b ions"), ("y", "y ions"), ("", "other peaks"))]
        legend = fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(theme["ink2"])
        fig.suptitle("Held-out spectra: every peak of the observed spectrum (top) and the prediction (bottom)",
                     fontsize=10, color=theme["ink"], x=0.07, ha="left")
        fig.text(0.07, 0.955, "Observed peaks are scaled to the base peak; each prediction is scaled so its "
                 "strongest ion is as high as the strongest observed b/y ion.\nColored: the b/y ions (fragment "
                 "charge 1-2) the models are trained and scored on. Gray: every other peak (isotopes, other ion "
                 "types, reporter ions, unassigned).", fontsize=7, color=theme["ink2"], ha="left", va="top")
        path = f"{out_prefix}_{mode}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=theme["surface"])
        plt.close(fig)
        print(f"wrote {path}", flush=True)


# ---- main -------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", required=True, help="held-out directory from split_heldout.py")
    ap.add_argument("--train", required=True, help="training directory from split_heldout.py (NCE calibration)")
    ap.add_argument("--model", action="append", default=[], help="Carafe model: 'generic', a .pt path, or name=path")
    ap.add_argument("--msp", required=True, help="the NIST MSP library (full observed spectra for the mirror plots)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calibration-spectra", type=int, default=1000, help="training spectra per instrument")
    ap.add_argument("--batch-size", type=int, default=1000, help="spectra per Koina request (the server maximum)")
    ap.add_argument("--workers", type=int, default=4, help="Koina requests in flight at once")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args(argv)
    gpu = args.device == "gpu"
    os.makedirs(args.out_dir, exist_ok=True)

    offsets = calibrate(args.train, args.calibration_spectra, args.batch_size, args.workers, gpu,
                        os.path.join(args.out_dir, "calibration.json"))

    psm, observed = load_heldout(args.heldout)
    in_range = in_prosit_range(psm)
    print(f"{in_range.sum()} of {len(psm)} held-out spectra are within Prosit's range "
          f"(length <= {MAX_LENGTH}, charge 1-{MAX_CHARGE}); scoring those", flush=True)
    sub_psm, sub_obs = reindex(psm[in_range], observed)
    instruments = sorted(sub_psm["instrument"].unique())
    missing = [i for i in instruments if i not in offsets]
    if missing:
        raise SystemExit(f"no calibrated offset for instrument(s) {missing}")

    # Prosit at the recorded and at the calibrated NCE (cached per set of offsets)
    settings = {"Prosit (recorded NCE)": {i: 0 for i in instruments},
                "Prosit (calibrated NCE: " + ", ".join(f"{i} {offsets[i]:+d}" for i in instruments) + ")":
                    {i: offsets[i] for i in instruments}}
    paths = {t: os.path.join(args.out_dir, "prosit_pred_" + "_".join(f"{i}{o:+d}" for i, o in s.items()) + ".tsv")
             for t, s in settings.items()}
    # cached predictions are only reused for the held-out set they were made for
    signature_path = os.path.join(args.out_dir, "prosit_pred_heldout.json")
    same_heldout = False
    if os.path.exists(signature_path):
        with open(signature_path) as fh:
            same_heldout = json.load(fh) == data_signature(args.heldout)
    todo = []
    for t in settings:  # (if all calibrated offsets are 0, both settings share one file: query it once)
        if not (same_heldout and os.path.exists(paths[t])) and paths[t] not in [paths[u] for u in todo]:
            todo.append(t)
    fresh = prosit_predict([(sub_psm, sub_psm["nce"] + sub_psm["instrument"].map(settings[t])) for t in todo],
                           [len(sub_obs)] * len(todo), args.batch_size, args.workers) if todo else []
    predictions = {}
    for t in settings:
        if t in todo:
            predictions[t] = fresh[todo.index(t)]
            predictions[t].to_csv(paths[t] + ".part", sep="\t", index=False, float_format="%.6g")
            os.replace(paths[t] + ".part", paths[t])
        else:
            predictions[t] = pd.read_csv(paths[t], sep="\t")
            if len(predictions[t]) != len(sub_obs):
                raise SystemExit(f"{paths[t]} does not match this held-out set; delete it to query Koina again")
            print(f"read {paths[t]}", flush=True)
    with open(signature_path, "w") as fh:
        json.dump(data_signature(args.heldout), fh)
    for spec in args.model:
        name, predicted = carafe_predict(spec, sub_psm, sub_obs, args.device)
        predictions[name] = predicted

    per_spectrum = {t: similarity(sub_psm, p, sub_obs, gpu) for t, p in predictions.items()}
    results = [summarize(t, r) for t, r in per_spectrum.items()]

    print(f"\n{'model':<44}{'spectra':>9}{'PCC':>8}{'COS':>8}{'SA':>8}{'SPC':>8}{'SA>0.9':>9}")
    for r in results:
        for label, s in ((r["model"], r), ("  default range", r["default_range"])):
            print(f"{label:<44}{s['spectra']:>9}{s['PCC']:>8.4f}{s['COS']:>8.4f}{s['SA']:>8.4f}"
                  f"{s['SPC']:>8.4f}{s['SA_gt_0.9']:>9.1%}")
    for r in results:
        print(f"\n{r['model']}: SA by instrument " + ", ".join(f"{k}={v:.4f}" for k, v in r["SA_by_instrument"].items())
              + " | by charge " + ", ".join(f"{k}+={v:.4f} ({n})" for k, (v, n) in r["SA_by_charge"].items()))
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
        json.dump({"heldout_spectra": int(len(psm)), "scored_spectra": int(in_range.sum()),
                   "nce_offsets": offsets, "results": results}, fh, indent=2)

    if args.model:
        carafe = list(predictions)[-1]
        calibrated = list(settings)[1]
        panels = {f"Carafe {carafe}": predictions[carafe], "Prosit, calibrated NCE": predictions[calibrated]}
        scores = {f"Carafe {carafe}": per_spectrum[carafe], "Prosit, calibrated NCE": per_spectrum[calibrated]}
        picks = pick_spectra(sub_psm)
        originals = original_spectra(args.msp, sub_psm, sub_obs, picks, os.path.join(args.out_dir, "mirror_spectra.json"))
        mirror_plots(sub_psm, originals, panels, scores, picks, os.path.join(args.out_dir, "mirror"))


if __name__ == "__main__":
    main()
