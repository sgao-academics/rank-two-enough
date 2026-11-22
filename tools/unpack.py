# -*- coding: utf-8 -*-
"""Unpack the stored datasets and evaluation payloads.

Every ``*.json.gz`` in this repository is a plain JSON file gzipped for
transport. This script writes the JSON back next to it, so the analysis scripts
-- which open ``.json`` paths -- run unchanged. The ``.gz`` files are never
modified.

    python tools/unpack.py                 # everything (about 1.1 GB unpacked)
    python tools/unpack.py pl4             # only files whose name contains pl4
    python tools/unpack.py --list          # show what would be written
    python tools/unpack.py --force         # rewrite targets that already exist

Unpacking is idempotent: an existing target is left alone unless --force is
given, so an interrupted run can simply be repeated.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH = ("data", "logs")
CHUNK = 2 ** 20


def find(patterns):
    out = []
    for top in SEARCH:
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base):
            continue
        for dp, _, fs in os.walk(base):
            for f in sorted(fs):
                if not f.endswith(".gz"):
                    continue
                if patterns and not any(p.lower() in f.lower() for p in patterns):
                    continue
                out.append(os.path.join(dp, f))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*",
                    help="substrings to match against file names")
    ap.add_argument("--force", action="store_true", help="overwrite existing targets")
    ap.add_argument("--list", action="store_true", help="list targets, write nothing")
    a = ap.parse_args()

    files = find(a.patterns)
    if not files:
        print("nothing matched %r under data/ or logs/" % (a.patterns,))
        return 1

    written = skipped = 0
    total = 0
    for src in files:
        dst = src[:-3]
        if os.path.exists(dst) and not a.force:
            skipped += 1
            continue
        if a.list:
            print("would write %s" % os.path.relpath(dst, ROOT))
            written += 1
            continue
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo, CHUNK)
        written += 1
        total += os.path.getsize(dst)

    if a.list:
        print("%d file(s) would be written, %d already present"
              % (written, skipped))
        return 0

    print("unpacked %d file(s), %.1f MB written, %d already present"
          % (written, total / 1048576.0, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
