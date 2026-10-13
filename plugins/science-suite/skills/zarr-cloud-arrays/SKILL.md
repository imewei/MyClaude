---
name: zarr-cloud-arrays
disable-model-invocation: true
description: Design, write, read, and migrate cloud-native Zarr stores (chunked N-D arrays on S3, GCS, Azure, or disk) with Python (zarr-python 3, Xarray, Dask) or Julia (Zarr.jl). Use for Zarr chunking, sharding, and layout, parallel writes, converting HDF5 (.h5, .hdf, .hdf5), NetCDF, or MATLAB v7.3 files to Zarr or cloud storage (including batches of large XPCS result files), Icechunk, VirtualiZarr or Kerchunk, and arrays too big for RAM, even if "Zarr" is never said.
---

# Cloud-Native Zarr Arrays

Zarr stores an N-D array as independently compressed **chunks** plus small JSON metadata. On object storage each chunk is one object, so readers fetch only the slice they need and many workers read or write at once. Most problems reduce to three decisions: chunk shape, who writes which chunk, and how metadata is found.

Lead agents: `python-pro` for Python, h5py, Xarray, and conversion batches; `julia-pro` for Zarr.jl. Delegate long multi-file jobs to them.

Load only what the task needs: [python.md](references/python.md) (zarr-python, Xarray, Dask, Icechunk), [julia.md](references/julia.md) (Zarr.jl), [hdf5.md](references/hdf5.md) (HDF5 or NetCDF conversion, many files), [xpcs.md](references/xpcs.md) (APS 8-ID-I XPCS results). APIs were run against zarr-python 3.4 and Zarr.jl 0.11 on 2026-10-13; recheck names on other versions.

## Format choice

Zarr for large N-D arrays with partial reads, parallel I/O, or object storage. Keep NetCDF/HDF5 for single files on shared POSIX disks. Use Virtual Zarr (VirtualiZarr, Kerchunk) when archives cannot be rewritten. Parquet suits tables, Cloud-Optimized GeoTIFF a single raster.

Prefer **v3** for new data (`zarr.json`, codec pipeline, **sharding**); write v2 only for readers that cannot read v3. Consolidated metadata (one request for the whole hierarchy) is a v2 feature; in v3 it is an extension outside the spec, so check the target reader.

## Design chunks before writing

Chunks are fixed once written; changing them rewrites the data.

1. **Name the dominant read** (point time series, map at one time, ML window). Make chunks long along what the read spans, short elsewhere. Opposing patterns need two copies.
2. **Size by bytes:** product of chunk dims × itemsize, roughly 1–100 MB uncompressed. Smaller means many tiny requests; larger wastes bandwidth. Compute it: `(100, 100, 1)` float32 is 40 KB, far too small.
3. **Shard for object count (v3):** shards of about 100 MB–1 GB keep a store at thousands of objects, not millions. The inner chunk is the read unit (a byte-range read), so inner chunks under 1 MB are fine inside large shards; size them to the read window.
4. **Align with source and writers:** use multiples of the source file's chunks, and give each writer whole chunks or shards.

**Codecs:** start with Blosc + Zstd level 3–5, shuffle on; time byte versus bit shuffle on a real slice. Lossy codecs change values, so the user must approve them.

## Writing safely

- Writing part of a chunk reads, modifies, and rewrites it; two workers on one chunk silently lose an update. Partition by chunk (or shard) boundaries.
- Create the empty array first (shape, dtype, chunks, `fill_value`), then fill regions in parallel.
- Plain object stores lack transactions, so a failed job leaves a visible half-written store. Use Icechunk, or write to a scratch prefix and switch after validation. Never overwrite the only copy.

## Reading efficiently

Open lazily (metadata, not data). Use consolidated metadata where supported and anonymous access for public buckets. A slice crossing N chunks costs N requests. Run compute in the bucket's region to avoid egress and slow transfers.

## Migrating HDF5 or NetCDF

Choose the route first (virtual Zarr versus rewrite, script versus Xarray): see [hdf5.md](references/hdf5.md). For a rewrite:

1. Inspect shapes, dtypes, internal chunks (`None` is contiguous), filters, fill values, scale/offset, attributes, dimension names.
2. Pick target chunks from the read pattern and byte target, as multiples of source chunks.
3. Create the empty target, then copy block by block; never load a dataset whole.
4. **Carry the missing-value marker:** set `fill_value` to the source's, because unwritten chunks read as `fill_value` (0 by default) and 0 looks like data. Keep CF `_FillValue` for Xarray. CF-masking libraries return `missing` or NaN, which mismatches the on-disk dtype; copy raw values or decode on purpose.
5. Copy attributes, dimension names, groups. List anything unrepresentable (references, non-numeric compounds); never drop it silently.

`scripts/h5_to_zarr.py` does this for HDF5, including many files in parallel and resumable (run `--help`; execute it, do not read its source). `scripts/stack_across_files.py` stacks one result across converted files. For 60-file batches, trial one file first (time, `peak_rss_mb`, object count) before scaling.

## Verify

Done means: nothing skipped (or the user accepted the list); shape, dtype, chunks, fill value, attributes match the plan; a full comparison (small input) or sampled slices (large) are equal, with tolerance only for approved lossy codecs; a reduction (sum, mean) agrees; object count and size are in range. Report the numbers, and keep the source until a reader has used the new store.

## Credentials and cost

Take credentials from the environment, a profile, or an instance role; never write or print keys. Listing and requests cost money, so check object counts and egress before a large job and flag plans producing millions of objects.
