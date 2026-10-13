#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr>=3.1", "numpy"]
# ///
"""Stack one dataset from every per-file group of a converted store into a single array with a leading file axis.

Run it after h5_to_zarr.py has written one group per file. Typical use is comparing a result such as g2 across scans:

  uv run stack_across_files.py all.zarr --path xpcs/multitau/normalized_g2 --path xpcs/multitau/delay_list \
      --sort-by entry/start_time

Output arrays go to <store>/combined/<path> (change with --out). Files whose array is missing or has a different
shape or dtype from the majority are listed and left out, so the result is never a silent mix; the exit code is 2
unless --allow-mismatch is given. The source groups are not modified. Rerunning replaces the combined arrays.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import zarr


def sort_key(g, path, fallback):
    """Value used to order files: first element of an array at `path` (bytes decoded), else the group name."""
    if path and path in g:
        v = np.asarray(g[path][...]).ravel()
        if v.size:
            x = v[0]
            return x.decode() if isinstance(x, bytes) else str(x) if v.dtype.kind in "OSUT" else float(x)
    return fallback


def blocks(shape, block):
    grid = [-(-n // b) for n, b in zip(shape, block)]
    for idx in np.ndindex(*grid):
        yield tuple(slice(i * b, min((i + 1) * b, n)) for i, b, n in zip(idx, block, shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--path", action="append", required=True, help="array path inside each file group (repeatable)")
    ap.add_argument("--sort-by", help="array path whose first value orders the files (for example entry/start_time)")
    ap.add_argument("--out", default="combined", help="group that receives the stacked arrays (default: combined)")
    ap.add_argument("--allow-mismatch", action="store_true", help="exit 0 even if some files were left out")
    a = ap.parse_args()

    root = zarr.open_group(a.store, mode="a")
    fmt = root.metadata.zarr_format
    names = [k for k, v in root.members() if isinstance(v, zarr.Group) and k != a.out and v.attrs.get("h5_to_zarr_complete")]
    if not names:
        sys.exit("no completed per-file groups found; run h5_to_zarr.py with several input files first")
    if a.out in root:
        del root[a.out]
    out = root.require_group(a.out)
    report, left_out = [], False
    for path in a.path:
        have = {n: root[n][path] for n in names if path in root[n]}
        missing = [n for n in names if n not in have]
        if not have:
            report.append({"path": path, "error": "not found in any file"})
            left_out = True
            continue
        common = Counter((arr.shape, str(arr.dtype)) for arr in have.values()).most_common(1)[0][0]
        use = sorted((n for n, arr in have.items() if (arr.shape, str(arr.dtype)) == common),
                     key=lambda n: sort_key(root[n], a.sort_by, n))
        differing = {n: [list(arr.shape), str(arr.dtype)] for n, arr in have.items() if n not in use}
        left_out = left_out or bool(missing or differing)
        shape, dtype = common[0], np.dtype(common[1]) if common[1] != "StringDType()" else str
        src0 = have[use[0]]
        chunks = (1,) + tuple(src0.chunks)
        kw = {"dimension_names": ("file",) + tuple(f"dim_{i}" for i in range(len(shape)))} if fmt == 3 else {}
        parent, _, leaf = path.rpartition("/")
        grp = out.require_group(parent) if parent else out
        z = grp.create_array(leaf, shape=(len(use),) + shape, dtype=dtype, chunks=chunks, fill_value=src0.fill_value if dtype is not str else None, **kw)
        bad = 0
        for i, n in enumerate(use):
            src = have[n]
            for sl in blocks(shape, tuple(src.chunks)) if shape else [()]:
                data = np.asarray(src[sl] if sl != () else src[...])
                z[(i,) + sl] = data
                back = np.asarray(z[(i,) + sl])
                bad += int(back.shape != data.shape or back.tolist() != data.tolist() if dtype is str else back.tobytes() != data.tobytes())
        z.attrs.update({"files": use, "sort_by": a.sort_by or "group name",
                        "sort_values": [sort_key(root[n], a.sort_by, n) for n in use]})
        report.append({"path": f"{a.out}/{path}", "shape": list(z.shape), "files": len(use), "missing_in": missing,
                       "left_out_shape_or_dtype": differing, "verify_mismatched_blocks": bad})
        if bad:
            left_out = True
    print(json.dumps({"store": a.store, "arrays": report}, indent=1))
    if any(r.get("verify_mismatched_blocks") for r in report):
        sys.exit(1)
    if left_out and not a.allow_mismatch:
        print("some files were left out or a path was not found; see the report, then rerun with --allow-mismatch to accept", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
