#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["h5py", "zarr>=3.1", "numpy"]
# ///
"""Convert one or many HDF5 files into a Zarr store, block by block, with verification.

  uv run h5_to_zarr.py --inspect a.h5 b.h5                       # plan only: sizes, object counts, unsupported items
  uv run h5_to_zarr.py a.h5 -o out.zarr                          # one file into the store root
  uv run h5_to_zarr.py runs/*.hdf -o all.zarr --jobs 8           # many files: one group per file, in parallel
  uv run h5_to_zarr.py runs/*.hdf -o all.zarr --shard-mb 256 --stack /xpcs/twotime/correlation_map

Memory is bounded by one output block (a chunk, or a shard with --shard-mb). Each file writes only its own
group, so parallel workers never touch the same chunk, and a rerun skips files already marked complete.
Strings (fixed or variable length) become Zarr UTF-8 strings, and compound datasets are converted (one array per field). Object
references and other unsupported items are listed in the report and make the exit code 2 unless --allow-skip.
Exit codes: 0 ok, 1 error or verification mismatch, 2 items skipped.
"""
import argparse
import json
import math
import os
import re
import resource
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import h5py
import numpy as np
import zarr

# HDF5 dimension-scale bookkeeping holds object references that have no Zarr equivalent.
SKIP_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME", "_Netcdf4Dimid", "_Netcdf4Coordinates", "_NCProperties"}
DONE = "h5_to_zarr_complete"


def to_json(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [to_json(x) for x in v.tolist()] if v.dtype.kind in "OSU" else v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    raise TypeError(f"unsupported attribute type {type(v).__name__}")


def copy_attrs(src, dst, where, skipped):
    out = {}
    for k, v in src.attrs.items():
        if k in SKIP_ATTRS:
            continue
        try:
            out[k] = to_json(v)
        except TypeError as e:
            skipped.append({"path": f"{where}@{k}", "reason": str(e)})
    dst.attrs.update(out)


def kind_of(dt):
    """'num', 'text', 'compound', or None when the dtype cannot be represented.

    HDF5 strings, fixed-length or variable-length, become Zarr UTF-8 strings. Fixed-length bytes have no stable
    Zarr v3 data type, and XPCS-style metadata (start_time, units, names) is text.
    """
    if h5py.check_string_dtype(dt) is not None:
        return "text"
    if dt.kind in "iufbc":
        return "num"
    if dt.kind == "V" and dt.names:
        ok = all(dt.fields[n][0].kind in "iufbc" for n in dt.names)
        return "compound" if ok else None
    return None


def pick_chunks(shape, itemsize, src_chunks, target_bytes):
    """Grow the source chunk (or one plane) by doubling axes, so output chunks stay multiples of source chunks."""
    if not shape:
        return ()
    if src_chunks:
        chunks = list(src_chunks)
    elif len(shape) > 1:
        chunks = [1] * (len(shape) - 2) + list(shape[-2:])
    else:
        chunks = list(shape)
    chunks = [max(1, min(c, n)) for c, n in zip(chunks, shape)]
    grown = True
    while grown and np.prod(chunks) * itemsize * 2 <= target_bytes:
        grown = False
        for ax in range(len(shape)):
            if chunks[ax] < shape[ax] and np.prod(chunks) * itemsize * 2 <= target_bytes:
                chunks[ax] = min(shape[ax], chunks[ax] * 2)
                grown = True
    return tuple(chunks)


def plan(shape, itemsize, src_chunks, a, name):
    """Chunk and shard shapes plus the object count they imply."""
    explicit = {p.lstrip("/"): tuple(int(x) for x in c.split(",")) for p, c in (s.split("=", 1) for s in a.chunks)}
    chunks = explicit.get(name) or pick_chunks(shape, itemsize, src_chunks, a.chunk_mb * 2**20)
    shards = pick_chunks(shape, itemsize, chunks, a.shard_mb * 2**20) if (a.shard_mb and shape and a.zarr_format == 3) else None
    if shards:  # a shard edge must be a multiple of the chunk edge, even where it was clipped to the array shape
        shards = tuple(-(-sh // c) * c for sh, c in zip(shards, chunks))
    unit = shards or chunks
    objects = int(np.prod([math.ceil(n / u) for n, u in zip(shape, unit)])) if shape else 1
    return chunks, shards, objects


def reader(d: h5py.Dataset, text: bool) -> Callable[[Any], Any]:
    """Slice reader for one dataset; `()` selects a scalar, strings decode to Python objects."""
    src = d.asstr() if text else d

    def read(sl: Any) -> Any:
        value = src[sl] if sl != () else src[()]
        return np.asarray(value, dtype=object) if text else value

    return read


def dim_names(ds):
    names = [(list(ds.dims[i].keys()) or [None])[0] for i in range(ds.ndim)]
    return names if any(names) else None


def blocks(shape, block):
    grid = [-(-n // b) for n, b in zip(shape, block)]
    for idx in np.ndindex(*grid):
        yield tuple(slice(i * b, min((i + 1) * b, n)) for i, b, n in zip(idx, block, shape))


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OTU" or b.dtype.kind in "OTU":
        return a.shape == b.shape and a.ravel().tolist() == b.ravel().tolist()
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def collect(f):
    """Names only. Holding Dataset objects open would keep a raw-chunk cache alive for every one of them."""
    groups, dsets = [], []

    def visit(name, obj):
        (groups if isinstance(obj, h5py.Group) else dsets).append(name)

    f.visititems(visit)
    return groups, dsets


def stackable(f, path):
    """Child datasets of an HDF5 group that share shape and dtype (so they can become one array), else None."""
    g = f.get(path.strip("/"))
    if not isinstance(g, h5py.Group):
        return None
    kids = sorted((n for n, o in g.items() if isinstance(o, h5py.Dataset)), key=natural)
    if len(kids) < 2:
        return None
    first = g[kids[0]]
    if kind_of(first.dtype) != "num" or any(g[k].shape != first.shape or g[k].dtype != first.dtype for k in kids):
        return None
    return kids


def inspect_file(src, a):
    with h5py.File(src, "r") as f:
        groups, dsets = collect(f)
        info = {"file": os.path.basename(src), "size_bytes": os.path.getsize(src), "groups": len(groups), "datasets": len(dsets),
                "uncompressed_bytes": 0, "objects_estimate": 0, "small_datasets": 0, "text_datasets": 0, "compound": 0,
                "unsupported": [], "stackable_groups": {}, "largest": []}
        sizes = []
        for name in dsets:
            d = f[name]
            nbytes = int(np.prod(d.shape or (1,))) * d.dtype.itemsize
            info["uncompressed_bytes"] += nbytes
            k = kind_of(d.dtype)
            if k is None:
                info["unsupported"].append({"path": name, "dtype": str(d.dtype)})
                continue
            info["text_datasets"] += k == "text"
            info["compound"] += k == "compound"
            info["small_datasets"] += nbytes < 65536
            _, _, objs = plan(d.shape, max(d.dtype.itemsize, 1), d.chunks, a, name)
            info["objects_estimate"] += objs * (len(d.dtype.names) if k == "compound" else 1) + 1
            sizes.append((nbytes, name, d.shape, str(d.dtype), d.chunks))
        parents = {}
        for name in dsets:
            parents.setdefault(name.rpartition("/")[0], []).append(name)
        for p, kids in parents.items():
            if p and len(kids) >= 8 and (s := stackable(f, p)):
                info["stackable_groups"][p] = len(s)
        info["objects_estimate"] += len(groups) + 1
        info["largest"] = [{"path": n, "mb": round(b / 2**20, 2), "shape": list(s), "dtype": t, "src_chunks": c}
                           for b, n, s, t, c in sorted(sizes, reverse=True)[:8]]
    return info


def make_array(grp, leaf, shape, dtype, chunks, shards, fill, a, names=None):
    kw = {}
    if a.zarr_format == 3:
        if dtype is not str:
            kw["compressors"] = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        if shards:
            kw["shards"] = shards
        if names:
            kw["dimension_names"] = names
    z = grp.create_array(leaf, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill, **kw)
    if a.zarr_format == 2 and names:
        z.attrs["_ARRAY_DIMENSIONS"] = [n or f"dim_{i}" for i, n in enumerate(names)]
    return z


def copy_blocks(z_list, shape, step, read, verify, mismatches, name):
    """Write every block of one source dataset to all its target arrays (fields of a compound), verifying as we go."""
    n = 0
    if not shape:
        data = read(())
        for z, field in z_list:
            payload = data if field is None else data[field]
            z[...] = payload
            if verify and not same(z[...], payload):
                mismatches.append({"path": name, "block": "scalar"})
        return 1
    for sl in blocks(shape, step):
        data = read(sl)
        for z, field in z_list:
            payload = data if field is None else data[field]
            z[sl] = payload
            if verify and not same(z[sl], payload):
                mismatches.append({"path": name + (f".{field}" if field else ""), "block": str(sl)})
        n += 1
    return n


def convert_file(src, dst, group, a):
    t0 = time.time()
    res = {"file": os.path.basename(src), "group": group, "status": "ok", "datasets": [], "skipped": [], "mismatches": [],
           "objects_estimate": 0, "src_bytes": os.path.getsize(src) if os.path.exists(src) else 0, "seconds": 0.0, "peak_rss_mb": 0}
    try:
        root = zarr.open_group(dst, mode="a", zarr_format=a.zarr_format)
        if group:
            if group in root and root[group].attrs.get(DONE) and not a.force:
                res["status"] = "skipped-complete"
                return res
            tgt = root.create_group(group, overwrite=True)
        else:
            tgt = root
        skipped, verify = res["skipped"], not a.no_verify
        # rdcc_nslots is allocated for every opened dataset, so keep it small (1M slots cost 8 MB each)
        with h5py.File(src, "r", rdcc_nbytes=int(a.rdcc_mb * 2**20), rdcc_nslots=10_007) as f:
            copy_attrs(f, tgt, "/", skipped)
            groups, dsets = collect(f)
            if not dsets:
                skipped.append({"path": "/", "reason": "file contains no datasets (empty or truncated?)"})
            stacks = {}
            for s in a.stack:
                if (kids := stackable(f, s)):
                    stacks[s.strip("/")] = kids
            stacked = {f"{p}/{k}" for p, kids in stacks.items() for k in kids}
            for name in groups:
                copy_attrs(f[name], tgt.require_group(name), name, skipped)
            for p, kids in stacks.items():
                h5g = f[p]
                first = h5g[kids[0]]
                shape = (len(kids),) + first.shape
                src_chunks = ((1,) + first.chunks) if first.chunks else None
                chunks, shards, objs = plan(shape, first.dtype.itemsize, src_chunks, a, p + "/stack")
                grp = tgt.require_group(p)
                z = make_array(grp, "stack", shape, first.dtype, chunks, shards, first.fillvalue, a)
                grp.attrs["members"] = kids
                grp.attrs["member_attrs"] = {k: {kk: to_json(vv) for kk, vv in h5g[k].attrs.items() if kk not in SKIP_ATTRS} for k in kids}

                def read(sl, h5g=h5g, kids=kids, first=first):
                    out = np.empty([len(range(*s.indices(n))) for s, n in zip(sl, (len(kids),) + first.shape)], first.dtype)
                    for j, i in enumerate(range(*sl[0].indices(len(kids)))):
                        out[j] = h5g[kids[i]][sl[1:]]
                    return out

                nblk = copy_blocks([(z, None)], shape, shards or chunks, read, verify, res["mismatches"], p + "/stack")
                res["objects_estimate"] += objs + 1
                res["datasets"].append({"path": p + "/stack", "shape": list(shape), "dtype": str(first.dtype), "chunks": list(z.chunks),
                                        "shards": list(shards or []), "blocks": nblk, "stacked_from": len(kids)})
            for name in dsets:
                if name in stacked:
                    continue
                d = f[name]
                kind = kind_of(d.dtype)
                if kind is None:
                    skipped.append({"path": name, "reason": f"unsupported dtype {d.dtype}"})
                    continue
                parent, _, leaf = name.rpartition("/")
                grp = tgt.require_group(parent) if parent else tgt
                itemsize = max(d.dtype.itemsize, 16 if kind == "text" else 1)
                chunks, shards, objs = plan(d.shape, itemsize, d.chunks, a, name)
                names = dim_names(d) if kind == "num" else None
                if kind == "num":
                    fill = d.fillvalue
                    z_list = [(make_array(grp, leaf, d.shape, d.dtype, chunks, shards, fill, a, names), None)]
                    read = reader(d, text=False)
                    owner = z_list[0][0]
                elif kind == "text":
                    z_list = [(make_array(grp, leaf, d.shape, str, chunks, shards, None, a), None)]
                    read = reader(d, text=True)
                    owner = z_list[0][0]
                else:  # compound: one array per field, grouped under the dataset's name
                    owner = grp.require_group(leaf)
                    z_list = [(make_array(owner, fld, d.shape, d.dtype.fields[fld][0], chunks, shards, None, a), fld)
                              for fld in d.dtype.names]
                    read = reader(d, text=False)
                    objs *= len(d.dtype.names)
                copy_attrs(d, owner, name, skipped)
                nblk = copy_blocks(z_list, d.shape, shards or chunks, read, verify, res["mismatches"], name)
                res["objects_estimate"] += objs + 1
                res["datasets"].append({"path": name, "kind": kind, "shape": list(d.shape), "dtype": str(d.dtype), "src_chunks": d.chunks,
                                        "chunks": list(chunks), "shards": list(shards or []), "blocks": nblk})
        res["objects_estimate"] += len(groups) + 1
        if not res["mismatches"]:
            tgt.attrs.update({"source_file": os.path.basename(src), "source_bytes": res["src_bytes"],
                              "source_mtime": os.path.getmtime(src), DONE: not (skipped and not a.allow_skip)})
    except Exception as e:  # noqa: BLE001  report and continue with the other files
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {e}"
    res["seconds"] = round(time.time() - t0, 2)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10))
    if res["mismatches"]:
        res["status"] = "mismatch"
    return res


def brief(r):
    """Compact per-file view for stdout; the full detail goes to --report."""
    keep = {k: r[k] for k in ("file", "group", "status", "seconds", "peak_rss_mb", "objects_estimate", "src_bytes", "error") if k in r}
    return keep | {"datasets": len(r["datasets"]), "skipped_total": len(r["skipped"]), "skipped": r["skipped"][:10],
                   "mismatch_total": len(r["mismatches"]), "mismatches": r["mismatches"][:5]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="+", help="HDF5 file(s)")
    ap.add_argument("-o", "--output", help="Zarr store path or URL (required unless --inspect)")
    ap.add_argument("--inspect", action="store_true", help="plan only; write nothing")
    ap.add_argument("--chunk-mb", type=float, default=8.0, help="target uncompressed chunk size (default 8)")
    ap.add_argument("--shard-mb", type=float, default=None, help="Zarr v3: group chunks into shards of about this size")
    ap.add_argument("--zarr-format", type=int, choices=(2, 3), default=3)
    ap.add_argument("--chunks", action="append", default=[], metavar="PATH=a,b,c", help="explicit chunks for one dataset (repeatable)")
    ap.add_argument("--stack", action="append", default=[], metavar="GROUP", help="stack same-shape datasets of an HDF5 group into one array (repeatable)")
    ap.add_argument("--group-per-file", action="store_true", help="one group per input file (automatic with several files)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes, one file each")
    ap.add_argument("--rdcc-mb", type=float, default=64.0, help="HDF5 raw chunk cache per file, MB")
    ap.add_argument("--force", action="store_true", help="redo files already marked complete")
    ap.add_argument("--no-verify", action="store_true", help="skip the block-by-block re-read comparison")
    ap.add_argument("--allow-skip", action="store_true", help="exit 0 even if items were skipped")
    ap.add_argument("--report", help="write the full JSON report (every dataset) to this path")
    ap.add_argument("--progress", action="store_true", help="print one line per finished file to stderr (automatic on a terminal)")
    ap.add_argument("--verbose", action="store_true", help="print the full report to stdout instead of the compact summary")
    a = ap.parse_args()
    if a.shard_mb and a.zarr_format != 3:
        ap.error("--shard-mb needs --zarr-format 3")
    if not a.inspect and not a.output:
        ap.error("-o/--output is required")

    if a.inspect:
        out = [inspect_file(s, a) for s in a.src]
        if a.report:
            with open(a.report, "w") as fh:
                json.dump(out, fh, indent=1, default=str)
        if a.verbose:
            print(json.dumps(out, indent=1, default=str))
            return
        cols = ("file", "size_bytes", "datasets", "small_datasets", "text_datasets", "compound", "objects_estimate", "stackable_groups")
        print(json.dumps({"files": len(out), "total_bytes": sum(i["size_bytes"] for i in out),
                          "objects_estimate_total": sum(i["objects_estimate"] for i in out),
                          "unsupported": [{"file": i["file"], "items": i["unsupported"][:5]} for i in out if i["unsupported"]],
                          "first_files": [{c: i[c] for c in cols} for i in out[:5]],
                          "largest_in_first_file": out[0]["largest"][:5]}, indent=1, default=str))
        return

    per_file = a.group_per_file or len(a.src) > 1
    groups = [os.path.splitext(os.path.basename(s))[0] if per_file else "" for s in a.src]
    if per_file and len(set(groups)) != len(groups):
        ap.error("input file names are not unique; rename them or run in separate stores")
    zarr.open_group(a.output, mode="a", zarr_format=a.zarr_format)  # create the root once, before workers start
    if a.jobs > 1 and len(a.src) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(convert_file, a.src, [a.output] * len(a.src), groups, [a] * len(a.src)))
    else:
        results = [convert_file(s, a.output, g, a) for s, g in zip(a.src, groups)]
    if a.progress or sys.stderr.isatty():
        for r in results:
            print(f"{r['file']}: {r['status']} {r['seconds']}s, peak {r['peak_rss_mb']} MB, {len(r['datasets'])} datasets, "
                  f"{len(r['skipped'])} skipped, ~{r['objects_estimate']} objects", file=sys.stderr)
    summary = {"files": len(results), "ok": sum(r["status"] in ("ok", "skipped-complete") for r in results),
               "objects_estimate_total": sum(r["objects_estimate"] for r in results),
               "src_bytes_total": sum(r["src_bytes"] for r in results), "results": results}
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
    if a.verbose:
        shown = summary
    else:  # aggregate plus only the files that need attention, so a 60-file run stays a few lines
        shown = {k: v for k, v in summary.items() if k != "results"} | {
            "max_peak_rss_mb": max((r["peak_rss_mb"] for r in results), default=0),
            "seconds_total": round(sum(r["seconds"] for r in results), 1),
            "needs_attention": [brief(r) for r in results if r["status"] not in ("ok", "skipped-complete") or r["skipped"] or r["mismatches"]],
            "skipped_complete": sum(r["status"] == "skipped-complete" for r in results)}
    print(json.dumps(shown, indent=1, default=str))
    if any(r["status"] in ("error", "mismatch") for r in results):
        sys.exit(1)
    if any(r["skipped"] for r in results) and not a.allow_skip:
        print("items were skipped; see the report, then rerun with --allow-skip to accept", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
