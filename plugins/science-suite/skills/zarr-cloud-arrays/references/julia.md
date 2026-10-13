# Zarr in Julia

Verified with Julia 1.13.1, Zarr.jl 0.11.1, NCDatasets 0.14.15, and DiskArrays 0.4.25 on 2026-10-13. Run Julia work through [julia-language](../../julia-language/SKILL.md) conventions: a project environment, no global installs.

## Contents
- Package map
- Zarr.jl basics
- Cloud stores
- Higher-level interfaces
- NetCDF to Zarr (tested)
- HDF5 sources
- Zarrs.jl and Icechunk status

## Package map

| Package | Role |
|---------|------|
| `Zarr.jl` | Pure-Julia Zarr v2 and v3 arrays and groups. Mature. A monorepo whose subpackages (`ZarrCore`, `ZarrBlosc`, `ZarrS3`, `ZarrGCS`, `ZarrHTTP`, `ZarrZip`) add codecs and stores. |
| `DiskArrays.jl` | Lets chunked arrays behave like `AbstractArray`; I/O happens when you index or broadcast. |
| `ZarrDatasets.jl` | Reads Zarr through the `CommonDataModel.jl` interface, so it behaves like NetCDF/GRIB datasets. |
| `YAXArrays.jl` | Labeled data cubes with `open_dataset` and `savecube(cube, "out.zarr"; driver=:zarr)`. |
| `NCDatasets.jl` / `HDF5.jl` | Source readers for migration. |
| `STAC.jl` | Query STAC catalogs to discover Zarr assets. |
| `Zarrs.jl` | Experimental Rust-backed implementation. See the last section. |

## Zarr.jl basics

`path` is a keyword. The shape is positional and follows the element type.

```julia
using Zarr

z = zcreate(Float32, 1000, 1000; path = "data.zarr", chunks = (100, 100),
            compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 3, shuffle = true))
z[1:100, 1:100] .= 1f0
z2 = zopen("data.zarr")       # lazy; add "r" for read-only
z2[1:10, 1:10]
```

**Dimension order is reversed from Python.** Zarr.jl shapes and `chunks` are in Julia (column-major) order, so a file stored as `(time, lat, lon)` is `(lon, lat, time)` in Julia, and `chunks = (12, 60, 80)` here means 80 time steps per chunk when Python reads it as `(time, lat, lon)`. Write the chunks you want in Python order, reverse them, and confirm with `zopen(...).metadata.chunks` or a Python `zarr.open_array(...).chunks` before copying data.

Use one index per dimension (`z[:, :]`). Chunk size is `prod(chunks) * sizeof(T)`: `(100, 100)` of `Float32` is 40 KB, so scale it to the 1–100 MB target in the main skill. Zarr.jl accepts `zarr_format = 3` and supports sharding; check the current docs for the exact keywords before relying on a sharded layout.

## Cloud stores

```julia
using Zarr
Zarr.ZarrS3.register!()                       # enable s3:// URLs (needs the S3 dependency loaded)
g = zopen("s3://bucket/data.zarr")            # public or credentialed via the environment
g = zopen("gs://cmip6/.../tas/gn/"; consolidated = true)   # one request for the hierarchy
```

Pass `consolidated = true` for stores that have consolidated metadata; it removes dozens of small requests when opening a deep hierarchy. Credentials come from the environment or an AWS profile, never literals.

## Higher-level interfaces

`ZarrDatasets.jl` plus `CommonDataModel.jl` give NetCDF-like access (variables, attributes, dimension names). `YAXArrays.jl` is the choice for combining many NetCDF files into one cube and saving it:

```julia
using YAXArrays, Zarr
ds = open_dataset("file.nc")
savecube(ds, "cube.zarr"; driver = :zarr)     # confirm the call against the YAXArrays version you install
```

Use this for large file collections; use the manual copy below when you need control over chunks, fill values, and attributes.

## NetCDF to Zarr (tested)

Two details differ from many online examples. `zcreate` takes `path` as a keyword, and an `NCDatasets` variable applies CF masking, so its element type is `Union{Missing, Float32}`, which is not a valid Zarr dtype. Copy from the raw `.var` and carry the fill value.

```julia
using NCDatasets, Zarr, DiskArrays

ds  = NCDataset("source.nc")
nc  = ds["temperature"]
raw = nc.var                              # raw storage values, no `missing`
T   = eltype(raw)

z = zcreate(T, size(raw)...; path = "target.zarr",
            chunks = (20, 15, 1),         # choose from the read pattern and byte target
            compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 3, shuffle = true),
            fill_value = nc.attrib["_FillValue"],
            attrs = Dict("units" => nc.attrib["units"]))

z .= raw                                  # DiskArrays copies chunk by chunk
close(ds)
```

Check the result: reopen with `zopen`, compare to the source `.var` (this run returned `true` for a 40×30×12 file), and compare `z.attrs`, `z.metadata.chunks`, and a reduction on larger inputs. Copy every attribute you need (`merge` the `nc.attrib` entries you want into `attrs`); a missing `_FillValue` or `units` is easy to lose. For a source with `scale_factor`/`add_offset`, decide whether to store packed integers with those attributes or decoded floats.

Match the target chunks to the source's internal chunking where you can. Writing one Zarr chunk from several HDF5 chunks forces repeated decompression.

## HDF5 sources

`HDF5.Dataset` is not a `DiskArrays` array in the tested versions, so `z .= dataset` does not stream by chunk. Copy block by block (tested; the result matched h5py with no transposing):

```julia
using HDF5, Zarr

function copy_blocks!(dst, src, blk)
    for I in CartesianIndices(map((n, b) -> 1:b:n, size(src), blk))
        rng = map((i, b, n) -> i:min(i + b - 1, n), Tuple(I), blk, size(src))
        dst[rng...] = src[rng...]
    end
    return dst
end

h5open("in.h5") do f
    d = f["obs/temp"]
    chunks = HDF5.get_chunk(d)           # source chunks, already in Julia (reversed) dimension order
    z = zcreate(eltype(d), size(d)...; path = "out.zarr", chunks,
                compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 3, shuffle = true),
                fill_value = eltype(d)(-9999),              # the dataset's fill value
                attrs = Dict("units" => read_attribute(d, "units")))
    copy_blocks!(z, d, chunks)
end
```

Dimensions are reversed relative to h5py and Python zarr: a `(60, 40, 30)` dataset is `(30, 40, 60)` in Julia. Fill values, dimension scales, and other attributes are not copied automatically; read and set them as above. For many files or HDF5 with strings and compound data, running `scripts/h5_to_zarr.py` is simpler.

## Zarrs.jl and Icechunk status

`Zarrs.jl` (github.com/earth-mover/Zarrs.jl) wraps the Rust `zarrs` library and supports Zarr v2 and v3, sharding, S3/GCS access, and Icechunk. Its own README labels it **experimental** and says to use Zarr.jl for mature work. It needs a Rust toolchain when built from source. When this was checked, `Zarrs` was not listed in the Julia General registry, although the README shows `Pkg.add("Zarrs")`. Confirm with `Pkg.add("Zarrs")` in a throwaway environment, and fall back to `Pkg.add(url = "https://github.com/earth-mover/Zarrs.jl")`.

There is no separate `Icechunk.jl` in General; Julia access to Icechunk goes through Zarrs.jl. Describe Zarrs.jl to users as promising and experimental, not as a drop-in replacement for Zarr.jl.
