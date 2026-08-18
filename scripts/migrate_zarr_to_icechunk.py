#!/usr/bin/env python3
"""Copy an existing plain-zarr ensemble store into an icechunk repository.

Ensembles are icechunk now (see ``generate_ensemble.py``), and the reader only opens
icechunk, so the stores that are still load-bearing have to come across: the southern-Africa
M=400 reference the scorecard rewrite is checked against, its matched M=100 control, and the
global k=5 hindcast pair.

``--verify`` re-reads both stores tile by tile and asserts the int16 arrays are identical.
That matters more than it sounds: the scorecard is about to be re-run on a migrated store to
prove a *validator* rewrite did not move any rows, and without a byte-level check on the copy
a difference would be ambiguous between the two changes.

    scripts/migrate_zarr_to_icechunk.py --src old.zarr --dst new_ic [--verify] [--delete_src]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.aggregate import ARRAY_NAME  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="existing plain-zarr store")
    ap.add_argument("--dst", required=True, help="icechunk repository to create")
    ap.add_argument("--verify", action="store_true",
                    help="re-read both stores and assert the arrays are identical")
    ap.add_argument("--verify_only", action="store_true",
                    help="skip the copy; only compare an existing pair")
    ap.add_argument("--delete_src", action="store_true",
                    help="remove the source store once the copy verifies")
    ap.add_argument("--budget_gb", type=float, default=4.0,
                    help="working set for one copied block")
    return ap.parse_args(argv)


def _block_plan(shape, chunks, budget_bytes):
    """Members and rows per copied block, on the chunk grid, inside the budget."""
    M, nH, H, W = shape
    mc = int(chunks[0]) if len(chunks) > 0 else 1
    rc = int(chunks[2]) if len(chunks) > 2 else 1024
    per = mc * rc * W * 2.0
    n_mc = max(1, int(budget_bytes // max(per, 1)))
    return min(M, n_mc * mc), min(H, rc)


def main(argv=None):
    args = parse_args(argv)
    import icechunk
    import zarr

    src = zarr.open(str(args.src), mode="r")
    shape, chunks = tuple(src.shape), tuple(src.chunks)
    attrs = dict(src.attrs)
    budget = args.budget_gb * 1e9
    m_step, r_step = _block_plan(shape, chunks, budget)
    M, nH, H, W = shape
    print(f"{args.src}  shape={shape} chunks={chunks} dtype={src.dtype}")
    print(f"  copying in blocks of {m_step} members x {r_step} rows")

    if not args.verify_only:
        dst_path = Path(args.dst)
        if dst_path.exists() and any(dst_path.iterdir()):
            raise SystemExit(f"{dst_path} already exists and is not empty")
        repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(dst_path)))
        session = repo.writable_session("main")
        root = zarr.create_group(session.store)
        dst = root.create_array(ARRAY_NAME, shape=shape, chunks=chunks, dtype=src.dtype,
                                fill_value=src.fill_value)
        dst.attrs.update(attrs)
        t0 = time.time()
        for m0 in range(0, M, m_step):
            m1 = min(m0 + m_step, M)
            for h in range(nH):
                for r0 in range(0, H, r_step):
                    r1 = min(r0 + r_step, H)
                    dst[m0:m1, h, r0:r1] = src[m0:m1, h, r0:r1]
            print(f"  members {m0}-{m1 - 1} in {time.time() - t0:.0f}s", flush=True)
        snap = session.commit(f"migrated from {args.src}")
        print(f"  committed {snap} in {(time.time() - t0) / 60:.1f} min")

    if args.verify or args.verify_only:
        from src.ensemble.aggregate import open_ensemble

        got, got_attrs = open_ensemble(args.dst)
        assert tuple(got.shape) == shape, f"shape {got.shape} != {shape}"
        assert tuple(got.chunks) == chunks, f"chunks {got.chunks} != {chunks}"
        missing = {k: v for k, v in attrs.items() if got_attrs.get(k) != v}
        assert not missing, f"attrs differ: {sorted(missing)}"
        n_diff = 0
        t0 = time.time()
        for m0 in range(0, M, m_step):
            m1 = min(m0 + m_step, M)
            for h in range(nH):
                for r0 in range(0, H, r_step):
                    r1 = min(r0 + r_step, H)
                    a = np.asarray(src[m0:m1, h, r0:r1])
                    b = np.asarray(got[m0:m1, h, r0:r1])
                    n_diff += int((a != b).sum())
            print(f"  verified members {m0}-{m1 - 1}, {n_diff} differing values "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if n_diff:
            raise SystemExit(f"✗ {n_diff:,} values differ — not deleting anything")
        print(f"✓ byte-identical over all {M * nH * H * W:,} values")

    if args.delete_src:
        if not (args.verify or args.verify_only):
            raise SystemExit("--delete_src requires --verify")
        shutil.rmtree(args.src)
        print(f"  removed {args.src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
