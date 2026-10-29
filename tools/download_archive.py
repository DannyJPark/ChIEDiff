#!/usr/bin/env python3
"""Fetch the data and weights archive from Zenodo and verify it.

    python3 tools/download_archive.py --doi 10.5281/zenodo.23029438

The DOI is a parameter, not a constant: the archive gets new versions, and hardcoding one
here would quietly hand a reproducer the wrong version. README.md names the current
one. Files land in archive/ (gitignored) and are checked against the archive's own SHA256SUMS,
which is one of the files in the record.

Stdlib only, so this runs before any environment exists.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--doi", required=True,
                   help="Zenodo DOI of the data/weights record, e.g. 10.5281/zenodo.1234567")
    p.add_argument("--out", default=str(ROOT / "archive"))
    p.add_argument("--no-extract", action="store_true",
                   help="download and verify only, leaving the tarballs packed")
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


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def present(dest: Path, f: dict) -> bool:
    """True when dest is the record's file: same size and, when the record gives one, the same
    checksum. A file left from an earlier version of the record has the same name and often the
    same size, so the size alone is not enough."""
    if not dest.is_file() or not f.get("size") or dest.stat().st_size != f["size"]:
        return False
    algo, _, value = (f.get("checksum") or "").partition(":")
    return not value or file_hash(dest, algo) == value


def download(url: str, dest: Path, size: int | None, give_up_after: int = 6) -> None:
    """Download to dest.part and resume it with a Range request whenever the connection drops,
    so an interrupted run, or a re-run, carries on from where it stopped. Gives up after
    give_up_after attempts in a row that add nothing, leaving the .part for the next run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    stalled = 0
    while stalled < give_up_after:
        have = tmp.stat().st_size if tmp.is_file() else 0
        if size and have >= size:
            break
        start = have
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0                          # the server sent the whole file again
                with tmp.open("ab" if have else "wb") as out:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if size:
                            pct = 100.0 * have / size
                            print(f"\r    {have / 1e6:7.1f} / {size / 1e6:.1f} MB  {pct:5.1f}%",
                                  end="", flush=True)
                        else:
                            print(f"\r    {have / 1e6:7.1f} MB", end="", flush=True)
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429) and e.code < 500:
                sys.exit(f"\n{dest.name}: HTTP {e.code} from Zenodo")
            print(f"\n    HTTP {e.code} at {have / 1e6:.1f} MB, retrying", flush=True)
        except (OSError, http.client.HTTPException) as e:
            print(f"\n    connection lost at {have / 1e6:.1f} MB ({type(e).__name__}), resuming",
                  flush=True)
        else:
            if size and have < size:
                print(f"\n    connection closed at {have / 1e6:.1f} MB, resuming", flush=True)
        if not size or have >= size:
            break
        stalled = 0 if have > start else stalled + 1
        time.sleep(min(60, 5 * (stalled + 1)))
    print()
    got = tmp.stat().st_size if tmp.is_file() else 0
    if size and got != size:
        sys.exit(f"{dest.name}: {got} of {size} bytes so far. Re-run the same command to resume.")
    tmp.replace(dest)


def shown(path: Path) -> str:
    """The path relative to the repository when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def unpack(out: Path) -> None:
    """Extract the archive's tarballs where the pipeline expects to find them.

    The receptors go to data/, because every worker resolves a manifest path relative to
    the repository root, and the measurements stay beside the archive.

    A tarball is skipped only when every file in it is already in place with its size, so a
    tarball from a newer version of the record is extracted over an older one.

    A member with an absolute path, a parent reference or a link is refused rather than
    written.
    """
    import tarfile

    targets = {"chiediff_receptors.tar.gz": ROOT / "data",
               "chiediff_measurements.tar.gz": out / "measurements"}
    for name, dest in targets.items():
        src = out / name
        if not src.is_file():
            continue
        with tarfile.open(src) as tar:
            members = []
            for m in tar.getmembers():
                p = Path(m.name)
                if p.is_absolute() or ".." in p.parts:
                    sys.exit(f"{name} contains an unsafe path: {m.name}")
                if m.issym() or m.islnk():
                    sys.exit(f"{name} contains a link, which is not expected: {m.name}")
                members.append(m)
            if all((dest / m.name).is_file() and (dest / m.name).stat().st_size == m.size
                   for m in members if m.isfile()):
                print(f"  {name}: already extracted in {shown(dest)}")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  {name}: extracting {len(members)} members to "
                  f"{shown(dest)}")
            # the 'data' filter where this Python has it, which also silences the warning
            # Python 3.12+ and patched 3.9 builds print without one
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(dest, members=members, **kw)


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

    out = Path(args.out).resolve()
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
        if present(dest, f):
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
            got = file_hash(target, "sha256")
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

    if not args.no_extract:
        unpack(out)

    print(f"\nArchive in {out}. It is gitignored. See docs/reproduce.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
