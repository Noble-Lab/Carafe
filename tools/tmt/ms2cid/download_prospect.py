#!/usr/bin/env python
"""Download PROSPECT PTMs: TMT (Zenodo record 8221499) and check every file's MD5.

The file list and checksums come from Zenodo's API and are compared with prospect_checksums.txt in
this folder, so a changed record is noticed. Files are fetched in parallel (one connection each is
slow; several together are much faster), a partial file is resumed, and a file whose MD5 does not
match is fetched again once.

    python download_prospect.py --out-dir prospect [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RECORD = "8221499"
API = f"https://zenodo.org/api/records/{RECORD}"
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS = os.path.join(HERE, "prospect_checksums.txt")


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_files():
    """{name: (md5, size)} from prospect_checksums.txt, checked against the live record."""
    recorded = {}
    with open(CHECKSUMS) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                checksum, size, name = line.split()
                recorded[name] = (checksum, int(size))
    with urllib.request.urlopen(API, timeout=120) as response:
        live = {f["key"]: (f["checksum"].split(":", 1)[1], int(f["size"])) for f in json.load(response)["files"]}
    if live != recorded:
        sys.exit(f"Zenodo record {RECORD} no longer matches {CHECKSUMS}; check what changed before using it")
    return recorded


def fetch(name, checksum, out_dir):
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and md5(path) == checksum:
        return f"{name}: already complete"
    url = f"https://zenodo.org/records/{RECORD}/files/{name}?download=1"
    for attempt in (1, 2):
        # curl resumes a partial file (-C -), retries transient errors, and writes no error page (--fail)
        subprocess.run(["curl", "-s", "-L", "--fail", "--retry", "5", "--retry-delay", "20", "-C", "-", "-o", path,
                        url], check=False)
        if os.path.exists(path) and md5(path) == checksum:
            return f"{name}: downloaded, MD5 OK"
        if os.path.exists(path):
            os.remove(path)  # corrupt: start again from nothing
    raise SystemExit(f"{name}: MD5 mismatch after two attempts")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    files = expected_files()
    print(f"{len(files)} files, {sum(s for _, s in files.values()) / 1e9:.1f} GB", flush=True)
    # largest first, so the long downloads start early
    order = sorted(files, key=lambda n: -files[n][1])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(lambda n: fetch(n, files[n][0], args.out_dir), order):
            print(message, flush=True)
    print("all files present and MD5-checked", flush=True)


if __name__ == "__main__":
    main()
