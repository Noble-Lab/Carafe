#!/usr/bin/env python
"""Score MS2 models on the held-out test set written by split_heldout.py.

Uses Carafe's own src/main/resources/py/v2/models.py for loading, prediction and the similarity
metrics, so the numbers are computed the same way as ai.py's test metrics. Each spectrum is
predicted at its own NCE and instrument (the nce and instrument columns of psm_pdv.txt).

    python evaluate_heldout.py --heldout split/heldout \
        --model generic --model base=../../../models/ms2tmt.pt --model sweep/B/ms2_model.pt

A model is "generic" (the installed AlphaPeptDeep model), or a .pt file, optionally named with
name=path. Prints and (with --out) writes, per model: median PCC, COS, SA and SPC and the share of
spectra with SA > 0.9, over all held-out spectra and over those inside Carafe's default library
range (length 7-35, precursor charge 2-4); and median SA per NCE, instrument, precursor charge and
length range.

--nce predicts every spectrum at that one collision energy instead of its own, to measure how much
a wrong NCE costs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS_PY_DIR = os.path.join(REPO_ROOT, "src", "main", "resources", "py", "v2")


def load_heldout(directory):
    import pandas as pd
    psm = pd.read_csv(os.path.join(directory, "psm_pdv.txt"), sep="\t", dtype={"mods": str, "mod_sites": str})
    psm["mods"] = psm["mods"].fillna("").astype(str)
    psm["mod_sites"] = psm["mod_sites"].fillna("").astype(str)
    psm["nAA"] = psm["sequence"].str.len()
    intensity = pd.read_csv(os.path.join(directory, "fragment_intensity_df.tsv"), sep="\t")
    return psm, intensity


def score(model_spec, psm, intensity, device, fixed_nce=None):
    sys.path.insert(0, MODELS_PY_DIR)
    from models import ModelManager, calc_ms2_similarity_mask

    name, _, path = model_spec.partition("=") if "=" in model_spec else (model_spec, "", model_spec)
    manager = ModelManager(mask_modloss=True, device=device)
    manager.load_installed_models("generic", model_list=["ms2"])
    if path != "generic":
        # models.load_external_models catches load errors and only prints a warning, which would
        # leave the generic model in place and score it under this model's name. Refuse instead.
        import torch
        if not os.path.isfile(path):
            raise SystemExit(f"{name}: model file not found: {path}")
        before = [p.detach().clone() for p in manager.ms2_model.model.parameters()]
        manager.load_external_models(ms2_model_file=path)
        after = list(manager.ms2_model.model.parameters())
        if all(torch.equal(b, a.detach()) for b, a in zip(before, after)):
            raise SystemExit(f"{name}: {path} did not load (the weights are still the generic model's)")
    frag = intensity[[c for c in manager.ms2_model.charged_frag_types if c in intensity.columns]].copy()
    for column in manager.ms2_model.charged_frag_types:
        if column not in frag.columns:
            frag[column] = 0.0
    frag = frag[manager.ms2_model.charged_frag_types]

    query = psm.copy()
    if fixed_nce is not None:
        query["nce"] = fixed_nce
        name = f"{name}@nce{fixed_nce:g}"
    start = time.time()
    predicted = manager.ms2_model.predict(query, reference_frag_df=frag)
    result, _ = calc_ms2_similarity_mask(psm.copy(), predicted, frag, None)
    elapsed = time.time() - start
    # calc_ms2_similarity_mask appends four summary rows (index PCC, COS, SA, SPC; all NaN) to the
    # per-spectrum table. Keep spectra only, or the count and the SA > 0.9 share are off.
    result = result[result.index.isin(psm.index)]
    if len(result) != len(psm):
        raise SystemExit(f"{name}: scored {len(result)} spectra but the held-out set has {len(psm)}")

    def medians(part):
        return {"spectra": int(len(part)),
                "PCC": float(part["PCC"].median()), "COS": float(part["COS"].median()),
                "SA": float(part["SA"].median()), "SPC": float(part["SPC"].median()),
                "SA_gt_0.9": float((part["SA"] > 0.9).mean())}

    sys.path.insert(0, HERE)
    from split_heldout import in_default_range

    length = result["sequence"].str.len()
    length_bin = length.map(lambda n: "<7" if n < 7 else (">35" if n > 35 else "7-35"))
    summary = {
        "model": name,
        **medians(result),
        # Carafe's default library range (length 7-35, charge 2-4): the spectra a default library holds
        "default_range": medians(result[in_default_range(result)]),
        "SA_by_nce": {str(k): float(v) for k, v in result.groupby("nce")["SA"].median().items()},
        "SA_by_instrument": {str(k): float(v) for k, v in result.groupby("instrument")["SA"].median().items()},
        "SA_by_charge": {str(k): [float(v), int(n)] for k, (v, n) in
                         result.groupby(result["charge"].astype(int))["SA"].agg(["median", "size"]).iterrows()},
        "SA_by_length": {str(k): [float(v), int(n)] for k, (v, n) in
                         result.groupby(length_bin)["SA"].agg(["median", "size"]).iterrows()},
        "seconds": round(elapsed, 1),
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", required=True, help="held-out directory from split_heldout.py")
    ap.add_argument("--model", action="append", required=True, help="'generic', a .pt path, or name=path")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--nce", type=float, help="predict every spectrum at this NCE instead of its own")
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args(argv)

    psm, intensity = load_heldout(args.heldout)
    results = [score(spec, psm, intensity, args.device, args.nce) for spec in args.model]

    print(f"{'model':<24}{'spectra':>9}{'PCC':>8}{'COS':>8}{'SA':>8}{'SPC':>8}{'SA>0.9':>9}")
    for r in results:
        for label, s in ((r["model"], r), (r["model"] + " (default range)", r["default_range"])):
            print(f"{label:<24}{s['spectra']:>9}{s['PCC']:>8.4f}{s['COS']:>8.4f}{s['SA']:>8.4f}"
                  f"{s['SPC']:>8.4f}{s['SA_gt_0.9']:>9.1%}")
    for r in results:
        print(f"\n{r['model']}: SA by NCE " + ", ".join(f"{k}={v:.4f}" for k, v in r["SA_by_nce"].items())
              + " | by instrument " + ", ".join(f"{k}={v:.4f}" for k, v in r["SA_by_instrument"].items()))
        print(f"{r['model']}: SA by charge " + ", ".join(f"{k}+={v:.4f} ({n})" for k, (v, n) in r["SA_by_charge"].items())
              + " | by length " + ", ".join(f"{k}={v:.4f} ({n})" for k, (v, n) in r["SA_by_length"].items()))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
