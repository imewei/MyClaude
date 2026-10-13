# APS 8-ID-I XPCS results files to Zarr

Paths come from the pyXpcsViewer key map (`aps_8idi.py`) and reader (`hdf_reader.py`), github.com/AdvancedPhotonSource/pyXpcsViewer, read 2026-10-13. Shapes and sizes are inferred from how that code reads and writes files, and tests ran on synthetic files built from the same key map, not beamline data. Run `--inspect` on a real file before trusting any size or shape here. General conversion steps are in [hdf5.md](hdf5.md).

## Layout

| Group | Datasets | Notes |
|-------|----------|-------|
| `/entry` | `start_time`; `instrument/detector_1/{frame_time, count_time, x_pixel_size, y_pixel_size}`; `instrument/incident_beam/incident_energy`; `sample`, `user` | Small. Strings and `units` attributes are bytes. |
| `/xpcs/qmap` | `mask`, `dynamic_roi_map`, `static_roi_map`; `{dynamic,static}_v_list_dim0/1`; `*_num_pts`; `*_index_mapping`; `beam_center_x/y`, `detector_distance`, `pixel_size`; `map_names`, `map_units` | Small; copy whole. |
| `/xpcs/multitau` | `delay_list`, `normalized_g2`, `normalized_g2_err`, `unnormalized_G2`, `normalized_g2_partial` (+ `_err`, `_labels`), `config/{stride_frame, avg_frame}` | The g2 compared across scans. |
| `/xpcs/spatial_mean` | `intensity_vs_time` | Long along frames. |
| `/xpcs/temporal_mean` | `scattering_2d` (read as 1×N×M), `scattering_1d`, `scattering_1d_segments` | |
| `/xpcs/twotime` | `correlation_map/c2_NNNNN` (one per q bin), `normalized_g2`, `normalized_g2_segments`, `delay_list`, `processed_bins`, `config/*` | Usually the bulk of a large file; confirm with `--inspect`. |

## Conventions that affect conversion

- **Analysis type varies per file.** The reader treats a file as `Twotime` if `/xpcs/twotime/correlation_map` exists and `Multitau` if `/xpcs/multitau/normalized_g2` exists, either or both, and errors on neither. Some files will lack the two-time group; `--stack` then does nothing and the cross-file step reports who lacks a path.
- **Shapes are padded, not squeezed:** 1-D results are `(1, N)` rows, scalars may be `(1, 1)`. Keep them in the store for exact round trips; squeeze at analysis time.
- **Strings become Zarr text;** `units` attributes are decoded to `str`.
- **g2 shapes can differ between scans** because the delay axis depends on the stride and averaging settings. Stack only files that agree; the rest are listed.
- **Many small objects:** one dataset per q bin. `--stack /xpcs/twotime/correlation_map` makes one array per file and keeps names and attributes in `members` and `member_attrs`.

## Recipe

```bash
uv run scripts/h5_to_zarr.py --inspect /data/xpcs/cycle2/sample_0001_results.hdf
uv run scripts/h5_to_zarr.py /data/xpcs/cycle2/*.hdf -o xpcs.zarr \
    --stack /xpcs/twotime/correlation_map --shard-mb 128 --jobs 4 --report convert_report.json
uv run scripts/stack_across_files.py xpcs.zarr \
    --path xpcs/multitau/normalized_g2 --path xpcs/multitau/delay_list --sort-by entry/start_time
```

Each step exits with code 2 if something was skipped or a scan was left out of a stack; read the report before accepting. Rerun the converter to resume. Trial one file first and size `--jobs` and the upload from its time, `peak_rss_mb`, and object estimate.

## Reading back

```python
import zarr
store = zarr.open_group("xpcs.zarr", mode="r")
g2 = store["combined/xpcs/multitau/normalized_g2"]     # (scan, n_tau, n_q); g2.attrs["files"] lists scans in start_time order
maps = store["<scan>/xpcs/twotime/correlation_map/stack"]                       # (n_maps, T, T)
names = store["<scan>/xpcs/twotime/correlation_map"].attrs["members"]           # c2_NNNNN in the same order
```

The mapping from a c2 index to a q bin comes from `processed_bins` and the qmap lists; confirm it with beamline documentation before building analysis on it.

## XPCS checks and limits

- Each converted file has the same analysis groups as its source; g2's q axis matches the dynamic q bins; `delay_list` increases; `start_time` is present and unique across files.
- pyXpcsViewer and similar tools read HDF5 directly, and a Zarr store needs reader changes. Keep the HDF files as the record until the analysis code moves, and say so when proposing the conversion.
- Layouts change between software versions, and the key map lists a second file type (`aps_8idi`) beside `nexus`. Inspect real files from each period.
- Direct writes to an object store were not tested.
