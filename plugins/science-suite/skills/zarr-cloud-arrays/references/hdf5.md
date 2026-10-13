# HDF5 to Zarr

Tested 2026-10-13 with h5py and zarr-python 3.4 on synthetic files (chunked gzip, dimension scales, fill values, contiguous data, scalars, compound and string datasets, XPCS-shaped sets). Implementation: `scripts/h5_to_zarr.py`. Julia sources: see [julia.md](julia.md#hdf5-sources).

## Contents
- Route
- Concept map and limits
- The script
- Many files
- Reading HDF5 efficiently
- Related formats

## Route

| Situation | Route |
|-----------|-------|
| Files stay put, are chunked, and use filters with Zarr codecs (gzip/zlib, shuffle, Blosc, Zstd, LZ4) | **Virtual Zarr** (VirtualiZarr or Kerchunk): byte-range manifest, no copy. Inherits HDF5 chunking, and the files must stay unchanged. |
| New chunking or sharding, filters without Zarr codecs (LZF, SZIP, plugins), contiguous layout, or many files into one store | **Rewrite** with the script or Xarray |
| Conforming NetCDF4 | Xarray: `open_dataset(path, engine="h5netcdf")` then `to_zarr` (python.md) |
| Plain HDF5 or MATLAB v7.3 `.mat` | The script, or Xarray per group (`group=`, `phony_dims="sort"`) |

Check `ds.chunks`: `None` is contiguous, so a virtual store would make it one huge chunk. Rewrite those.

## Concept map and limits

| HDF5 | Zarr |
|------|------|
| Group, dataset | Group, array |
| Attribute | `attrs` (JSON only; arrays become lists; references dropped; dimension-scale bookkeeping not copied) |
| Chunks, filters | `chunks` (choose again), codecs |
| `fillvalue` | `fill_value` |
| Dimension scale | `dimension_names` (v3) or `_ARRAY_DIMENSIONS` attribute (v2, read by Xarray) |
| Strings, fixed or variable length | UTF-8 strings (fixed-length bytes have no stable v3 type) |
| Compound, numeric fields | One array per field in a group of the dataset's name |
| Hard-link aliases | Copied once, at the first path visited |

Not converted (listed in the report, exit code 2): compound types with non-numeric fields, object and region references, soft and external links, other dtypes. Enums copy as integers and lose names; store the mapping in an attribute. Resolve references yourself.

## The script

```bash
uv run scripts/h5_to_zarr.py --inspect a.h5 b.h5                 # plan only: sizes, object estimate, stackable groups
uv run scripts/h5_to_zarr.py a.h5 -o out.zarr                    # one file into the store root
uv run scripts/h5_to_zarr.py runs/*.hdf -o all.zarr --jobs 8 --shard-mb 128 --stack /group/with/many/maps
uv run scripts/h5_to_zarr.py a.h5 -o out.zarr --chunks obs/temp=720,30,30 --zarr-format 2 --allow-skip
```

- **Chunks follow the source:** doubling the source chunks keeps each output chunk a multiple of whole source chunks. Override per dataset with `--chunks`.
- **Bounded memory:** one output block at a time, about 100 MB plus 5–10× the block size (128 MB shards need about 1.3 GB per worker). Size `--jobs` from the reported `peak_rss_mb`.
- **`--stack GROUP`:** merges same-shape, same-dtype datasets (hundreds of `c2_00001`-style maps) into `GROUP/stack`, keeping `members` and `member_attrs` attributes. Thousands of small objects become a few large ones.
- **Parallel and resumable:** several inputs give one group per file (named from the file stem), so workers never share a chunk. Finished groups are marked complete and skipped on rerun (`--force` redoes). A failed file is reported and the rest continue.
- **Verified and quiet:** every block is re-read and compared (`--no-verify` skips). Stdout is an aggregate plus only the files needing attention; the full per-dataset report goes to `--report FILE` (`--verbose` prints it, `--progress` adds a line per file). Exit 0 ok, 1 error or mismatch, 2 items skipped.

Execute the script; do not read its source. Parallelism is by process because h5py serializes calls behind a global lock.

## Many files (for example 60 files of 1.2 GB)

1. **Inspect** two or three representative files. What dominates bytes and object count decides the layout; file size does not.
2. **Layout:** one store, one group per file (lossless, parallel, resumable). Stack small same-shape datasets with `--stack` and shard large arrays. Combine results across files in a second pass: `scripts/stack_across_files.py STORE --path <array> --sort-by <timestamp array>` writes `combined/<array>` with a leading file axis and lists files whose array is missing or differs in shape or dtype.
3. **Trial one full-size file:** record time, `peak_rss_mb`, objects. With only small samples, build a synthetic file with the real dataset shapes and sizes (scale the long axes up) and time that; small samples hide gzip decode and I/O cost. Total objects ≈ files × objects per file; workers ≈ min(cores, RAM ÷ peak memory); time ≈ per-file time × files ÷ workers, but a remote store will be the bottleneck, so measure a small upload.
4. **Run with `--jobs`**, then rerun the same command to confirm every file reports complete. Keep the report as the manifest.
5. **Verify independently:** compare sampled datasets with separate code, check attributes and strings on several files, and open the store with the analysts' reader.
6. **Upload:** convert to local scratch, then copy from a machine in the bucket's region (`aws s3 sync`, `rclone`). Direct `s3://` writes were not tested.

Measured on a laptop SSD: one synthetic 1 GB file (seventy 2000×2000 float32 gzip maps) took 4–7 s including verification, peaking at 0.5 GB (8 MB chunks), 0.7 GB (32 MB shards), 1.3 GB (128 MB shards). Real files and networks differ; run your own trial.

## Reading HDF5 efficiently

- HDF5 decompresses a whole chunk to serve any part of it, so align reads to source chunks.
- The chunk cache is per open dataset and each opened dataset allocates its slot table (1M slots cost 8 MB each). Close datasets when done and keep `rdcc_nslots` modest (`h5py.File(p, rdcc_nbytes=512 * 2**20, rdcc_nslots=10_007)`).
- Remote HDF5 issues many small range requests. For a one-time conversion, copy the file to local scratch in the same region. If reading remotely, use a block-caching fsspec file or the `ros3` driver (when built in), and time a small dataset first.
- Resizable datasets: use the current `ds.shape`; resize the Zarr array later if the source grows.

## Related formats

- **MATLAB v7.3 `.mat`** is HDF5, column-major (dimensions look reversed in h5py), strings as uint16, complex as compound (split into real and imaginary first).
- **HDF4 and HDF-EOS2** are a different format: convert to HDF5 or NetCDF4 (`h4toh5`, GDAL) or read with `pyhdf`, then continue here.
- **NetCDF4** is HDF5 with conventions; prefer Xarray when they hold.
