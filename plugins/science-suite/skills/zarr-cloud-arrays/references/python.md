# Zarr in Python

Verified with zarr-python 3.4.1, Xarray, and Dask on 2026-10-13. Install `zarr`; add `s3fs` or `gcsfs` for cloud (`zarr[remote]`), and `xarray` and `dask` for labeled data.

## Contents
- zarr-python 3: create, shard, compress
- Cloud stores
- Xarray: open, write, template and region
- Dask for parallel writes
- NetCDF and HDF5 to Zarr
- Virtual Zarr, Icechunk, GPU, ML loaders

## zarr-python 3: create, shard, compress

```python
import zarr

z = zarr.create_array(
    "data/a.zarr",
    shape=(10_000, 10_000),
    chunks=(500, 500),          # read granularity
    shards=(5_000, 5_000),      # object granularity (v3 only); must be a multiple of chunks
    dtype="f4",
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
    overwrite=True,
)
z[:500, :500] = 1.0
```

`create_array` takes `compressors` (plural). Older v2-era code uses `compressor` and `zarr.create`. For v2 output pass `zarr_format=2`. Inspect with `z.info_complete()`.

## Cloud stores

```python
group = zarr.open_group("s3://bucket/path.zarr", mode="r", storage_options={"anon": True})
z = zarr.create_array("s3://bucket/new", shape=(100, 100), chunks=(10, 10), dtype="f4")
```

An `s3://` or `gs://` URL creates an `FsspecStore` and needs `s3fs` or `gcsfs`. Pass credentials through `storage_options` sourced from the environment, never literals.

`zarr.consolidate_metadata(store)` writes consolidated metadata. For v3 it emits a `ZarrUserWarning` that this is outside the v3 spec; that is expected.

## Xarray

```python
import xarray as xr

ds = xr.open_zarr("s3://bucket/cube.zarr", chunks={}, storage_options={"anon": True})  # lazy, native chunks
sub = ds["tas"].sel(time=slice("2000", "2001")).isel(lat=slice(0, 90))
sub.mean("time").compute()
```

`chunks={}` keeps the store's chunking as the Dask chunking; `chunks=None` skips Dask entirely. Select before computing so only the needed chunks are fetched.

Set encoding at write time. Keys depend on the format and Xarray version: `compressors` for zarr 3, `compressor` for zarr 2.

```python
ds.to_zarr("out.zarr", mode="w", encoding={"tas": {"chunks": (1, 180, 360)}})
```

### Template then region writes (the pattern for large stores)

```python
template = ds_full.chunk({"time": 1})                 # lazy, with the final shape and chunks
template.to_zarr("out.zarr", compute=False, mode="w") # writes metadata only

for i in range(n_time):                               # or one task per time block
    part = compute_slice(i)                           # Dataset with a time dimension of length 1
    part.to_zarr("out.zarr", region={"time": slice(i, i + 1), "lat": slice(None), "lon": slice(None)})
```

Each region must cover whole chunks, and `part` must not carry variables or coordinates that fall outside the region (drop them first). Tested: 8 single-step writes into a template matched the source.

## Dask

`ds.to_zarr(...)` with a Dask-backed dataset writes chunks in parallel. Keep Dask chunks equal to, or a whole multiple of, the Zarr chunks (or shards). Dask chunks that straddle Zarr chunks make two tasks write one chunk, which corrupts data. Xarray raises an error for the common cases; do not silence it.

## NetCDF and HDF5 to Zarr

For plain HDF5 (groups, no coordinates), compound or contiguous data, or any case needing a verified block-by-block rewrite, use `scripts/h5_to_zarr.py` and see [hdf5.md](hdf5.md). The Xarray route below suits conforming NetCDF4.

Small and medium files:

```python
ds = xr.open_dataset("src.nc", chunks={"time": 1})   # lazy
ds.to_zarr("out.zarr", mode="w", encoding={v: {"chunks": (1, *ds[v].shape[1:])} for v in ds.data_vars})
```

Xarray decodes CF masking and scale factors by default, so integers packed with a scale become floats. Pass `mask_and_scale=False` to copy raw values. The Zarr `fill_value` defaults to 0 for integers, so unwritten chunks would read as 0; set it explicitly (for example `encoding={"t2m": {"fill_value": -32767}}` on a packed variable) or let Xarray write `_FillValue`, and check the result. For a huge or multi-file set, build the template from the first file and write each file as a region.

## Virtual Zarr, Icechunk, GPU, ML loaders

- **VirtualiZarr / Kerchunk** index existing NetCDF/HDF5/TIFF byte ranges into a manifest so they read like Zarr with no copy. Good when the archive cannot move or be rewritten. Reads are bounded by the original files' chunking.
- **Icechunk** adds transactions, branches, and snapshots on top of Zarr stores in object storage. Use it when several writers or readers share a store, or when you need to roll back. Follow its current documentation for the API; it is a separate package with its own release cycle.
- **kvikio** enables GPUDirect Storage reads into GPU memory for CuPy arrays. It needs matching NVIDIA hardware and drivers.
- **Xbatcher** slices Xarray/Zarr data into training batches, and the result feeds a PyTorch `DataLoader`. Set chunks to the batch window so each batch is one or a few requests. For JAX input pipelines see [jax-core-programming](../../jax-core-programming/SKILL.md).
