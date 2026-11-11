#!/usr/bin/env python3
"""Fetch reviewed Swiss-Prot entries for the Test 1 length arms from UniProt.

Uses the UniProt REST /search endpoint with cursor pagination (the /stream
endpoint materialises the full result server-side and times out on queries
of this size).

  small   : reviewed, 1-99 aa
  mid     : reviewed, 100-199 aa
  control : reviewed, 200-400 aa   (positive-control arm)

Output: data/raw/swissprot_{arm}.jsonl.gz, one JSON object per UniProt TSV row.
Pages are appended as they arrive and the cursor is checkpointed in
data/raw/swissprot_{arm}.state.json, so an interrupted run resumes.

Usage: python scripts/01_fetch_swissprot.py [--arm small|mid|control|all] [--force]

Standard library only.
"""

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data/raw"

FIELDS = ",".join([
    "accession", "id", "protein_name", "organism_name", "length",
    "sequence", "xref_pfam", "go_id", "ft_transmem", "protein_existence",
])

# Three contiguous length arms, so the length axis has no gap.
ARMS = {"small": (1, 99), "mid": (100, 199), "control": (200, 400)}
PAGE_SIZE = 500
LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def get(url, tries=6):
    """GET with retry/backoff. Returns (body_text, next_url, total)."""
    delay = 3
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                link = resp.headers.get("Link", "")
                total = resp.headers.get("x-total-results")
                m = LINK_NEXT.search(link)
                return body, (m.group(1) if m else None), total
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if attempt == tries:
                raise
            print(f"    retry {attempt}/{tries - 1} after {exc} (sleep {delay}s)", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)


def fetch_arm(arm, force=False):
    lo, hi = ARMS[arm]
    out = RAW / f"swissprot_{arm}.jsonl.gz"
    state_path = RAW / f"swissprot_{arm}.state.json"

    if out.exists() and not force and not state_path.exists():
        n = sum(1 for _ in gzip.open(out, "rt"))
        print(f"[skip] {out.name} complete ({n} entries)")
        return out

    query = f"reviewed:true AND length:[{lo} TO {hi}]"
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({
        "query": query, "fields": FIELDS, "format": "tsv", "size": PAGE_SIZE,
    })

    state = {}
    mode = "wt"
    if state_path.exists() and not force:
        state = json.loads(state_path.read_text())
        url = state.get("next")
        mode = "at"
        print(f"[resume] {arm} from page {state.get('pages', 0)} "
              f"({state.get('rows', 0)} rows already written)")
        if not url:
            print(f"[skip] {arm} already finished")
            state_path.unlink()
            return out

    header = None
    rows = state.get("rows", 0)
    pages = state.get("pages", 0)
    total = None
    t0 = time.time()

    with gzip.open(out, mode) as fh:
        while url:
            body, nxt, tot = get(url)
            if tot and total is None:
                total = int(tot)
                print(f"[fetch] {arm}: {lo}-{hi} aa, {total} entries", flush=True)

            lines = body.rstrip("\n").split("\n")
            if not lines or not lines[0]:
                break
            page_header = lines[0].split("\t")
            if header is None:
                header = page_header
            body_lines = lines[1:]

            for line in body_lines:
                parts = line.split("\t")
                if len(parts) != len(page_header):
                    continue
                fh.write(json.dumps(dict(zip(page_header, parts))) + "\n")
                rows += 1

            pages += 1
            url = nxt
            state_path.write_text(json.dumps({"next": url, "rows": rows, "pages": pages}))

            if pages % 20 == 0 or url is None:
                rate = rows / max(time.time() - t0, 1e-9)
                pct = f"{100 * rows / total:.0f}%" if total else "?"
                print(f"    page {pages:>4}  {rows:>7} rows  {pct}  {rate:.0f}/s", flush=True)

    state_path.unlink(missing_ok=True)
    print(f"[done]  {out.name}: {rows} entries, {out.stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    ap.add_argument("--force", action="store_true", help="ignore checkpoint and refetch")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        fetch_arm(arm, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
