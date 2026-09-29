#!/usr/bin/env python3
"""Fetch the data and weights archive from Zenodo and verify it.

    python3 tools/download_archive.py --doi 10.5281/zenodo.23029439

The DOI is a parameter, not a constant: the archive gets new versions, and hardcoding one
here would quietly hand a reproducer the wrong version. README section 2 names the current
one. Files land in archive/ (gitignored) and are checked against the archive's own SHA256SUMS,
which is one of the files in the record.

Stdlib only, so this runs before any environment exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--doi", required=True,
                   help="Zenodo DOI of the data/weights record, e.g. 10.5281/zenodo.1234567")
    p.add_argument("--out", default=str(ROOT / "archive"))
    p.add_argument("--only", nargs="*", metavar="NAME",
                   help="fetch only these filenames (default: all)")
    p.add_argument("--list", action="store_true", help="list the record's files and exit")
    return p.parse_args()


def record_id(doi: str) -> str:
    """Zenodo DOIs end in the record id."""
    tail = doi.rstrip("/").rsplit(".", 1)[-1]
    if not tail.isdigit():
        sys.exit(f"cannot read a record id out of {doi!r}; expected it to end in digits")
    return tail


def fetch_metadata(rid: str) -> dict:
    url = f"https://zenodo.org/api/records/{rid}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo returned HTTP {e.code} for record {rid}. Check the DOI.")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Zenodo: {e.reason}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if size:
                pct = 100.0 * done / size
                print(f"\r    {done / 1e6:7.1f} / {size / 1e6:.1f} MB  {pct:5.1f}%",
                      end="", flush=True)
            else:
                print(f"\r    {done / 1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    args = parse_args()
    rid = record_id(args.doi)
    meta = fetch_metadata(rid)

    title = (meta.get("metadata") or {}).get("title", "?")
    files = meta.get("files") or []
    print(f"record {rid}: {title}")
    print(f"  doi     : {meta.get('doi')}")
    print(f"  files   : {len(files)}, {sum(f.get('size', 0) for f in files) / 1e6:.1f} MB total")
    print()

    if args.list:
        for f in files:
            print(f"  {f.get('size', 0) / 1e6:8.1f} MB  {f.get('key')}")
        return 0

    out = Path(args.out)
    wanted = [f for f in files if not args.only or f.get("key") in args.only]
    if args.only and len(wanted) != len(args.only):
        missing = set(args.only) - {f.get("key") for f in files}
        sys.exit(f"not in the record: {sorted(missing)}")

    for f in wanted:
        key = f.get("key")
        dest = out / key
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if not link:
            print(f"  {key}: no download link in the record metadata; skipped")
            continue
        if dest.is_file() and f.get("size") and dest.stat().st_size == f["size"]:
            print(f"  {key}: already present, skipping")
            continue
        print(f"  {key}")
        download(link, dest, f.get("size"))

    # The record ships its own SHA256SUMS. Prefer it over Zenodo's per-file checksums: it is
    # what the authors computed at packing time, in the standard format, so `sha256sum -c`
    # works on it too.
    sums = out / "SHA256SUMS"
    print()
    if sums.is_file():
        print("verifying against the archive's SHA256SUMS")
        bad, checked = [], 0
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            target = out / name.strip()
            if not target.is_file():
                continue
            checked += 1
            got = sha256(target)
            state = "OK" if got == digest.strip() else "MISMATCH"
            print(f"  {state:8s} {name.strip()}")
            if state != "OK":
                bad.append(name.strip())
        if bad:
            print(f"\n{len(bad)} file(s) failed verification: {bad}")
            print("Delete them and re-run; a truncated download is the usual cause.")
            return 1
        print(f"\n{checked} file(s) verified.")
    else:
        print("SHA256SUMS not present, so the files were not verified.")

    print(f"\nArchive in {out}. It is gitignored. See README section 3, Tier 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
