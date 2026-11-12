"""Shared lookup for result tables, which may be stored as .tsv or .tsv.gz.

The search and kNN scripts write plain .tsv; the Zenodo archive holds the same
tables gzipped. Readers name a table either way and find whichever exists.
"""
import gzip
from pathlib import Path


def find_table(path):
    """The existing file among `path`, `path`.gz and `path` without .gz, or None."""
    p = Path(path)
    for c in (p, p.with_name(p.name + ".gz"),
              p.with_suffix("") if p.suffix == ".gz" else None):
        if c is not None and c.exists():
            return c
    return None


def open_table(path):
    """Open a table for text reading whichever form exists, or return None."""
    p = find_table(path)
    if p is None:
        return None
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
