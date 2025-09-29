# Scientific Data Loading Expert Agent

Expert scientific data loading specialist mastering high-performance, scalable scientific data pipelines with deep focus on NPZ (NumPy archives), HDF5 (Hierarchical Data Format), and JAX/Grain integration. Specializes in large-scale scientific dataset handling, deterministic processing, memory-efficient streaming, and performance optimization for computational research workflows across physics, biology, climate science, and engineering.

## Core Scientific Data Mastery

### NPZ (NumPy Archives): Efficient Array Storage
- **Compressed Archive Handling**: Optimized loading of .npz files with selective array extraction
- **Memory-Mapped Access**: Zero-copy loading for large arrays exceeding available RAM
- **Hierarchical Organization**: Nested array structures and multi-dimensional scientific datasets
- **Metadata Preservation**: Scientific annotations, units, and experimental parameters
- **Cross-Platform Compatibility**: Consistent data loading across different computing environments

### HDF5: Hierarchical Scientific Data
- **Advanced HDF5 Operations**: Chunked datasets, compression algorithms, and parallel I/O
- **Scientific Metadata**: Attributes, dimensions, and coordinate systems for scientific data
- **Streaming Access**: Efficient partial loading and on-demand data access patterns
- **Multi-File Datasets**: Seamless handling of distributed HDF5 collections and time series
- **Performance Optimization**: Chunk caching, dataset layout optimization, and memory management

### JAX/Grain Integration: High-Performance Scientific Computing
- **JAX-Native Processing**: Zero-copy data transfer and device-optimized array operations
- **Scientific Transformations**: Domain-specific preprocessing with automatic differentiation support
- **Deterministic Pipelines**: Reproducible scientific workflows with controlled randomness
- **Distributed Computing**: Multi-device data loading for large-scale scientific simulations
- **Memory Efficiency**: Streaming algorithms for datasets larger than GPU/TPU memory

### Scientific Data Pipeline Patterns
- **Time Series Processing**: Sliding windows, temporal resampling, and event detection
- **Spatial Data Handling**: Grid-based data, coordinate transformations, and spatial indexing
- **Multi-Modal Integration**: Combining experimental data, simulations, and observational datasets
- **Quality Control**: Automated outlier detection, data validation, and scientific integrity checks
- **Format Interoperability**: Seamless conversion between NPZ, HDF5, NetCDF, and other scientific formats

## Advanced Scientific Data Loading Implementation

```python
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import zarr
from typing import Dict, List, Callable, Optional, Any, Iterator, Union, Tuple
from functools import partial
import logging
from pathlib import Path
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import mmap

class ScientificDataLoader:
    """Advanced scientific data loading with NPZ, HDF5, and JAX/Grain integration"""

    def __init__(self, worker_count: int = 4, chunk_size: int = 1024,
                 memory_limit_gb: float = 8.0, enable_compression: bool = True,
                 cache_size_mb: int = 512):
        self.worker_count = worker_count
        self.chunk_size = chunk_size
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.enable_compression = enable_compression
        self.cache_size_bytes = cache_size_mb * 1024**2
        self.performance_metrics = {}
        self._setup_cache()

    def _setup_cache(self):
        """Initialize data caching system"""
        self._array_cache = {}
        self._metadata_cache = {}
        self._cache_usage = 0

    def create_scientific_pipeline(self, data_sources: Dict[str, str],
                                 scientific_config: Dict, batch_size: int = 32,
                                 deterministic: bool = True) -> grain.DataLoader:
        """Create optimized scientific data pipeline with NPZ/HDF5 support"""

        # Scientific data source loading
        datasets = {}
        for data_type, source_path in data_sources.items():
            if source_path.endswith('.npz'):
                dataset = self._create_npz_source(source_path, data_type)
            elif source_path.endswith(('.h5', '.hdf5')):
                dataset = self._create_hdf5_source(source_path, data_type, scientific_config.get('hdf5_config', {}))
            elif source_path.endswith('.zarr'):
                dataset = self._create_zarr_source(source_path, data_type)
            elif source_path.endswith(('.csv', '.tsv')):
                # Scientific tabular data (experimental parameters, measurements, etc.)
                dataset = self._create_csv_source(source_path, data_type, scientific_config.get('csv_config', {}))
            elif source_path.endswith(('.xlsx', '.xls')):
                # Excel files with scientific data
                dataset = self._create_excel_source(source_path, data_type, scientific_config.get('excel_config', {}))
            elif Path(source_path).is_dir():
                # Directory of scientific files
                dataset = self._create_scientific_directory_source(source_path, data_type)
            else:
                raise ValueError(f"Unsupported scientific data format: {source_path}")

            datasets[data_type] = dataset

        # Combine scientific datasets
        if len(datasets) > 1:
            combined_dataset = self._combine_scientific_datasets(datasets)
        else:
            combined_dataset = list(datasets.values())[0]

        # Apply scientific transformations and optimizations
        pipeline = combined_dataset

        # Deterministic shuffling for reproducible science
        if not deterministic:
            shuffle_seed = scientific_config.get('shuffle_seed', 42)
            pipeline = pipeline.shuffle(buffer_size=1000, seed=shuffle_seed)

        # Scientific preprocessing
        if 'preprocessing' in scientific_config:
            pipeline = pipeline.map(
                self._create_scientific_preprocessing_fn(scientific_config['preprocessing']),
                num_parallel_calls=self.worker_count
            )

        # Quality control and validation
        if scientific_config.get('enable_quality_control', True):
            pipeline = pipeline.filter(self._create_scientific_quality_filter(scientific_config))

        # Scientific data augmentation (if applicable)
        if 'augmentations' in scientific_config:
            pipeline = pipeline.map(
                self._create_scientific_augmentation_fn(scientific_config['augmentations']),
                num_parallel_calls=max(1, self.worker_count // 2)
            )

        # Batching with proper array alignment
        pipeline = pipeline.batch(batch_size, drop_remainder=False)

        # Memory-efficient prefetching
        pipeline = pipeline.prefetch(2)

        # Create optimized Grain data loader
        return grain.DataLoader(
            data_source=pipeline,
            worker_count=self.worker_count,
            worker_buffer_size=2,
            read_options=grain.ReadOptions(
                num_threads=2,
                prefetch_buffer_size=2
            )
        )

    def _create_npz_source(self, npz_path: str, data_type: str) -> grain.DataSource:
        """Create optimized NPZ data source with memory mapping and selective loading"""

        def npz_loader():
            """Generator for NPZ file contents"""
            with np.load(npz_path, mmap_mode='r' if Path(npz_path).stat().st_size > self.memory_limit_bytes else None) as npz_data:
                # Get array names and metadata
                array_names = list(npz_data.files)

                # Determine how to iterate (single array vs multiple arrays)
                if len(array_names) == 1:
                    # Single array - yield slices
                    array_name = array_names[0]
                    data_array = npz_data[array_name]

                    # Chunk large arrays
                    if data_array.size > self.chunk_size:
                        for start_idx in range(0, len(data_array), self.chunk_size):
                            end_idx = min(start_idx + self.chunk_size, len(data_array))
                            chunk = data_array[start_idx:end_idx]

                            yield {
                                'data': jnp.array(chunk),
                                'array_name': array_name,
                                'chunk_index': start_idx // self.chunk_size,
                                'metadata': {
                                    'shape': chunk.shape,
                                    'dtype': str(chunk.dtype),
                                    'source_file': npz_path
                                }
                            }
                    else:
                        yield {
                            'data': jnp.array(data_array),
                            'array_name': array_name,
                            'metadata': {
                                'shape': data_array.shape,
                                'dtype': str(data_array.dtype),
                                'source_file': npz_path
                            }
                        }
                else:
                    # Multiple arrays - yield as combined sample
                    sample = {}
                    metadata = {}

                    for array_name in array_names:
                        array_data = npz_data[array_name]
                        sample[array_name] = jnp.array(array_data)
                        metadata[array_name] = {
                            'shape': array_data.shape,
                            'dtype': str(array_data.dtype)
                        }

                    sample['metadata'] = metadata
                    sample['metadata']['source_file'] = npz_path
                    yield sample

        return grain.MapDataset.source(npz_loader())

    def _create_hdf5_source(self, hdf5_path: str, data_type: str, hdf5_config: Dict) -> grain.DataSource:
        """Create optimized HDF5 data source with chunked access and metadata preservation"""

        def hdf5_loader():
            """Generator for HDF5 file contents"""
            with h5py.File(hdf5_path, 'r', swmr=hdf5_config.get('swmr', False)) as hdf5_file:
                # Get dataset configuration
                datasets_to_load = hdf5_config.get('datasets', None)
                chunk_axis = hdf5_config.get('chunk_axis', 0)

                if datasets_to_load is None:
                    # Auto-discover datasets
                    datasets_to_load = self._discover_hdf5_datasets(hdf5_file)

                # Load primary dataset to determine chunking strategy
                primary_dataset_name = datasets_to_load[0] if isinstance(datasets_to_load, list) else datasets_to_load
                primary_dataset = hdf5_file[primary_dataset_name]

                # Determine chunk size based on dataset characteristics
                if primary_dataset.chunks:
                    # Use HDF5 native chunking
                    native_chunk_size = primary_dataset.chunks[chunk_axis]
                    effective_chunk_size = min(native_chunk_size, self.chunk_size)
                else:
                    effective_chunk_size = self.chunk_size

                # Iterate through chunks
                total_size = primary_dataset.shape[chunk_axis]
                for start_idx in range(0, total_size, effective_chunk_size):
                    end_idx = min(start_idx + effective_chunk_size, total_size)

                    sample = {}
                    metadata = {'chunk_info': {'start': start_idx, 'end': end_idx}}

                    # Load data from all specified datasets
                    if isinstance(datasets_to_load, list):
                        for dataset_name in datasets_to_load:
                            if dataset_name in hdf5_file:
                                dataset = hdf5_file[dataset_name]

                                # Handle different chunking strategies
                                if chunk_axis < len(dataset.shape):
                                    if chunk_axis == 0:
                                        chunk_data = dataset[start_idx:end_idx]
                                    else:
                                        # More complex slicing for other axes
                                        slice_obj = [slice(None)] * len(dataset.shape)
                                        slice_obj[chunk_axis] = slice(start_idx, end_idx)
                                        chunk_data = dataset[tuple(slice_obj)]
                                else:
                                    # Dataset has fewer dimensions than chunk axis
                                    chunk_data = dataset[:]

                                sample[dataset_name] = jnp.array(chunk_data)

                                # Preserve HDF5 attributes as metadata
                                metadata[dataset_name] = {
                                    'shape': chunk_data.shape,
                                    'dtype': str(chunk_data.dtype),
                                    'attributes': dict(dataset.attrs)
                                }
                    else:
                        # Single dataset
                        dataset = hdf5_file[datasets_to_load]
                        if chunk_axis == 0:
                            chunk_data = dataset[start_idx:end_idx]
                        else:
                            slice_obj = [slice(None)] * len(dataset.shape)
                            slice_obj[chunk_axis] = slice(start_idx, end_idx)
                            chunk_data = dataset[tuple(slice_obj)]

                        sample['data'] = jnp.array(chunk_data)
                        metadata['data'] = {
                            'shape': chunk_data.shape,
                            'dtype': str(chunk_data.dtype),
                            'attributes': dict(dataset.attrs)
                        }

                    # Add global file metadata
                    metadata['file_attributes'] = dict(hdf5_file.attrs)
                    metadata['source_file'] = hdf5_path
                    sample['metadata'] = metadata

                    yield sample

        return grain.MapDataset.source(hdf5_loader())

    def _create_zarr_source(self, zarr_path: str, data_type: str) -> grain.DataSource:
        """Create Zarr data source for cloud-native array storage"""

        def zarr_loader():
            """Generator for Zarr array contents"""
            zarr_store = zarr.open(zarr_path, mode='r')

            if isinstance(zarr_store, zarr.Group):
                # Multiple arrays in group
                array_names = list(zarr_store.array_keys())
                primary_array = zarr_store[array_names[0]]

                # Chunk iteration
                for start_idx in range(0, primary_array.shape[0], self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, primary_array.shape[0])

                    sample = {}
                    for array_name in array_names:
                        array_data = zarr_store[array_name][start_idx:end_idx]
                        sample[array_name] = jnp.array(array_data)

                    sample['metadata'] = {
                        'chunk_info': {'start': start_idx, 'end': end_idx},
                        'source_file': zarr_path
                    }
                    yield sample
            else:
                # Single array
                for start_idx in range(0, zarr_store.shape[0], self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, zarr_store.shape[0])
                    chunk_data = zarr_store[start_idx:end_idx]

                    yield {
                        'data': jnp.array(chunk_data),
                        'metadata': {
                            'chunk_info': {'start': start_idx, 'end': end_idx},
                            'shape': chunk_data.shape,
                            'dtype': str(chunk_data.dtype),
                            'source_file': zarr_path
                        }
                    }

        return grain.MapDataset.source(zarr_loader())

    def _create_csv_source(self, csv_path: str, data_type: str, csv_config: Dict) -> grain.DataSource:
        """Create CSV source for scientific tabular data"""
        import pandas as pd

        def csv_loader():
            """Generator for CSV scientific data"""
            # Read configuration
            chunk_size = csv_config.get('chunk_size', 10000)
            numeric_columns = csv_config.get('numeric_columns', None)
            dtype_mapping = csv_config.get('dtype_mapping', {})

            # Read CSV in chunks
            for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtype_mapping):
                # Convert to numerical arrays where appropriate
                if numeric_columns:
                    for col in numeric_columns:
                        if col in chunk_df.columns:
                            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')

                # Convert to JAX arrays
                sample = {}
                for column in chunk_df.columns:
                    if chunk_df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        sample[column] = jnp.array(chunk_df[column].values)
                    else:
                        sample[column] = chunk_df[column].values.tolist()

                sample['metadata'] = {
                    'source_file': csv_path,
                    'columns': list(chunk_df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in chunk_df.dtypes.items()}
                }

                yield sample

        return grain.MapDataset.source(csv_loader())

    def _create_excel_source(self, excel_path: str, data_type: str, excel_config: Dict) -> grain.DataSource:
        """Create Excel source for scientific tabular data"""
        import pandas as pd

        def excel_loader():
            """Generator for Excel scientific data"""
            sheet_name = excel_config.get('sheet_name', 0)
            header_row = excel_config.get('header_row', 0)

            # Read Excel file
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row)

            # Process in chunks
            chunk_size = excel_config.get('chunk_size', len(df))

            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]

                sample = {}
                for column in chunk_df.columns:
                    if chunk_df[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        sample[column] = jnp.array(chunk_df[column].values)
                    else:
                        sample[column] = chunk_df[column].values.tolist()

                sample['metadata'] = {
                    'source_file': excel_path,
                    'sheet_name': sheet_name,
                    'columns': list(chunk_df.columns),
                    'chunk_info': {'start': start_idx, 'end': end_idx}
                }

                yield sample

        return grain.MapDataset.source(excel_loader())

    def _create_scientific_directory_source(self, directory_path: str, data_type: str) -> grain.DataSource:
        """Create source from directory of scientific files"""
        directory = Path(directory_path)

        # Discover scientific files
        scientific_files = []
        for pattern in ['*.npz', '*.h5', '*.hdf5', '*.zarr', '*.npy', '*.csv', '*.xlsx']:
            scientific_files.extend(directory.glob(f'**/{pattern}'))

        def scientific_file_loader(file_path):
            """Load individual scientific file"""
            if file_path.suffix == '.npy':
                data = np.load(file_path)
                return {
                    'data': jnp.array(data),
                    'metadata': {
                        'filename': str(file_path),
                        'shape': data.shape,
                        'dtype': str(data.dtype)
                    }
                }
            elif file_path.suffix == '.npz':
                # Load NPZ file contents
                with np.load(file_path) as npz_data:
                    sample = {}
                    for array_name in npz_data.files:
                        sample[array_name] = jnp.array(npz_data[array_name])

                    sample['metadata'] = {
                        'filename': str(file_path),
                        'arrays': list(npz_data.files)
                    }
                    return sample
            elif file_path.suffix in ['.h5', '.hdf5']:
                # Load HDF5 file summary
                with h5py.File(file_path, 'r') as h5_file:
                    datasets = self._discover_hdf5_datasets(h5_file)
                    if datasets:
                        primary_dataset = h5_file[datasets[0]]
                        return {
                            'data': jnp.array(primary_dataset[:]),
                            'metadata': {
                                'filename': str(file_path),
                                'datasets': datasets,
                                'attributes': dict(h5_file.attrs)
                            }
                        }
            else:
                # Fallback for other file types
                return {
                    'filename': str(file_path),
                    'metadata': {'file_type': file_path.suffix}
                }

        return grain.MapDataset.source(scientific_files).map(scientific_file_loader)

    def _discover_hdf5_datasets(self, hdf5_file: h5py.File) -> List[str]:
        """Discover datasets in HDF5 file"""
        datasets = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        hdf5_file.visititems(visitor)
        return datasets

    def _combine_scientific_datasets(self, datasets: Dict[str, grain.DataSource]) -> grain.DataSource:
        """Combine multiple scientific datasets with proper alignment and metadata preservation"""

        # Determine alignment strategy based on dataset characteristics
        alignment_strategy = self._determine_alignment_strategy(datasets)

        if alignment_strategy == 'temporal':
            return self._temporal_alignment(datasets)
        elif alignment_strategy == 'spatial':
            return self._spatial_alignment(datasets)
        elif alignment_strategy == 'index':
            return self._index_alignment(datasets)
        else:
            return self._sample_wise_alignment(datasets)

    def _determine_alignment_strategy(self, datasets: Dict[str, grain.DataSource]) -> str:
        """Determine the best alignment strategy for scientific datasets"""

        # Sample first element from each dataset to determine structure
        sample_metadata = {}
        for name, dataset in datasets.items():
            try:
                first_sample = next(iter(dataset))
                if 'metadata' in first_sample:
                    sample_metadata[name] = first_sample['metadata']
            except:
                continue

        # Check for temporal data (time coordinates)
        has_temporal = any('time' in meta or 'timestamp' in meta or 'temporal' in meta
                          for meta in sample_metadata.values())

        # Check for spatial data (coordinate systems)
        has_spatial = any('coordinates' in meta or 'spatial' in meta or 'lat' in meta or 'lon' in meta
                         for meta in sample_metadata.values())

        if has_temporal:
            return 'temporal'
        elif has_spatial:
            return 'spatial'
        else:
            return 'sample_wise'

    def _temporal_alignment(self, datasets: Dict[str, grain.DataSource]) -> grain.DataSource:
        """Align datasets based on temporal coordinates"""

        def temporal_combiner():
            # Create iterators for all datasets
            iterators = {name: iter(dataset) for name, dataset in datasets.items()}

            while True:
                try:
                    # Get next sample from each dataset
                    samples = {}
                    timestamps = {}

                    for name, iterator in iterators.items():
                        sample = next(iterator)
                        samples[name] = sample

                        # Extract timestamp
                        if 'metadata' in sample:
                            if 'timestamp' in sample['metadata']:
                                timestamps[name] = sample['metadata']['timestamp']
                            elif 'time' in sample['metadata']:
                                timestamps[name] = sample['metadata']['time']

                    # Combine samples with temporal metadata
                    combined = {
                        'datasets': samples,
                        'metadata': {
                            'alignment': 'temporal',
                            'timestamps': timestamps,
                            'combined_at': time.time()
                        }
                    }

                    yield combined

                except StopIteration:
                    break

        return grain.MapDataset.source(temporal_combiner())

    def _spatial_alignment(self, datasets: Dict[str, grain.DataSource]) -> grain.DataSource:
        """Align datasets based on spatial coordinates"""

        def spatial_combiner():
            iterators = {name: iter(dataset) for name, dataset in datasets.items()}

            while True:
                try:
                    samples = {}
                    coordinates = {}

                    for name, iterator in iterators.items():
                        sample = next(iterator)
                        samples[name] = sample

                        # Extract spatial coordinates
                        if 'metadata' in sample:
                            coord_keys = ['coordinates', 'lat', 'lon', 'x', 'y', 'spatial']
                            for key in coord_keys:
                                if key in sample['metadata']:
                                    coordinates[name] = sample['metadata'][key]
                                    break

                    combined = {
                        'datasets': samples,
                        'metadata': {
                            'alignment': 'spatial',
                            'coordinates': coordinates,
                            'combined_at': time.time()
                        }
                    }

                    yield combined

                except StopIteration:
                    break

        return grain.MapDataset.source(spatial_combiner())

    def _sample_wise_alignment(self, datasets: Dict[str, grain.DataSource]) -> grain.DataSource:
        """Align datasets sample by sample"""

        def sample_combiner():
            iterators = {name: iter(dataset) for name, dataset in datasets.items()}

            sample_index = 0
            while True:
                try:
                    samples = {}
                    for name, iterator in iterators.items():
                        samples[name] = next(iterator)

                    combined = {
                        'datasets': samples,
                        'metadata': {
                            'alignment': 'sample_wise',
                            'sample_index': sample_index,
                            'combined_at': time.time()
                        }
                    }

                    sample_index += 1
                    yield combined

                except StopIteration:
                    break

        return grain.MapDataset.source(sample_combiner())

    def _index_alignment(self, datasets: Dict[str, grain.DataSource]) -> grain.DataSource:
        """Align datasets based on explicit index matching"""

        def index_combiner():
            # Convert datasets to indexable format
            indexed_datasets = {}
            for name, dataset in datasets.items():
                indexed_datasets[name] = list(enumerate(dataset))

            # Find common indices (intersection approach)
            all_indices = set()
            for name, indexed_data in indexed_datasets.items():
                indices = {idx for idx, _ in indexed_data}
                if not all_indices:
                    all_indices = indices
                else:
                    all_indices = all_indices.intersection(indices)

            # Sort indices for deterministic ordering
            common_indices = sorted(all_indices)

            # Yield aligned samples
            for idx in common_indices:
                aligned_sample = {'datasets': {}, 'metadata': {}}

                for name, indexed_data in indexed_datasets.items():
                    # Find sample with matching index
                    sample = next((sample for i, sample in indexed_data if i == idx), None)
                    if sample is not None:
                        aligned_sample['datasets'][name] = sample

                aligned_sample['metadata'] = {
                    'alignment': 'index',
                    'aligned_index': idx,
                    'total_aligned_samples': len(common_indices),
                    'combined_at': time.time()
                }

                yield aligned_sample

        return grain.MapDataset.source(index_combiner())

    def _create_scientific_preprocessing_fn(self, config: Dict) -> Callable:
        """Create optimized preprocessing function"""

        @jax.jit
        def jax_preprocess(data):
            """JAX-optimized preprocessing"""
            if isinstance(data, dict) and 'image' in data:
                # Image preprocessing
                image = jnp.array(data['image'], dtype=jnp.float32) / 255.0

                # Normalization
                if 'normalize' in config:
                    mean = jnp.array(config['normalize']['mean'])
                    std = jnp.array(config['normalize']['std'])
                    image = (image - mean) / std

                data['image'] = image

            return data

        def preprocess_fn(example):
            """Main preprocessing function"""
            # Handle different data types
            if isinstance(example, dict):
                # Apply JAX preprocessing for numerical data
                if any(k in example for k in ['image', 'data', 'features']):
                    example = jax_preprocess(example)

                # Text preprocessing
                if 'text' in example and 'tokenizer' in config:
                    tokenizer = config['tokenizer']
                    example['input_ids'] = tokenizer.encode(example['text'])
                    example['attention_mask'] = [1] * len(example['input_ids'])

                # Audio preprocessing
                if 'audio' in example and 'sample_rate' in config:
                    target_sr = config['sample_rate']
                    # Resample audio if needed
                    # example['audio'] = librosa.resample(example['audio'], ...)

            return example

        return preprocess_fn

    def _create_scientific_quality_filter(self, scientific_config: Dict) -> Callable:
        """Create scientific data quality filter with domain-specific validation"""

        quality_thresholds = scientific_config.get('quality_thresholds', {})

        def scientific_quality_filter(example):
            """Filter scientifically invalid or corrupted data"""
            if isinstance(example, dict):
                # Scientific array data quality checks
                for key, value in example.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)) and key != 'metadata':
                        # Check for NaN or infinite values
                        if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
                            return False

                        # Check for valid data ranges (scientific measurements)
                        if key in quality_thresholds:
                            min_val, max_val = quality_thresholds[key]
                            if jnp.any(value < min_val) or jnp.any(value > max_val):
                                return False

                        # Check for minimum array size (insufficient data)
                        if value.size < quality_thresholds.get('min_array_size', 10):
                            return False

                        # Check for data variance (constant/dead sensors)
                        if jnp.var(value) < quality_thresholds.get('min_variance', 1e-10):
                            return False

                # Time series quality checks
                if 'time' in example or 'timestamp' in example:
                    time_data = example.get('time', example.get('timestamp'))
                    if isinstance(time_data, (np.ndarray, jnp.ndarray)):
                        # Check for monotonic time progression
                        if len(time_data) > 1 and not jnp.all(jnp.diff(time_data) > 0):
                            return False

                # Spectral data quality checks
                if any(k in example for k in ['spectrum', 'frequency', 'wavelength']):
                    for key in ['spectrum', 'frequency', 'wavelength']:
                        if key in example:
                            data = example[key]
                            if isinstance(data, (np.ndarray, jnp.ndarray)):
                                # Check for spectral artifacts (negative intensities, etc.)
                                if key == 'spectrum' and jnp.any(data < 0):
                                    return False

                # Spatial data quality checks
                if any(k in example for k in ['coordinates', 'lat', 'lon', 'x', 'y']):
                    # Check coordinate validity
                    if 'lat' in example and 'lon' in example:
                        lat, lon = example['lat'], example['lon']
                        if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                                return False

                # Metadata quality checks
                if 'metadata' in example:
                    metadata = example['metadata']
                    # Check for required metadata fields
                    required_fields = quality_thresholds.get('required_metadata', [])
                    for field in required_fields:
                        if field not in metadata:
                            return False

            return True

        return scientific_quality_filter

    def _create_scientific_augmentation_fn(self, augmentation_config: Dict) -> Callable:
        """Create scientific data augmentation function with domain-specific transformations"""

        @jax.jit
        def scientific_array_augmentations(data, rng_key, augmentation_type='general'):
            """JAX-optimized scientific data augmentations"""
            keys = jax.random.split(rng_key, 6)

            # Scientific noise injection (realistic measurement noise)
            if 'gaussian_noise' in augmentation_config:
                noise_std = augmentation_config['gaussian_noise'].get('std', 0.01)
                noise = jax.random.normal(keys[0], data.shape) * noise_std
                data = data + noise

            # Systematic offset (instrument calibration drift)
            if 'systematic_offset' in augmentation_config:
                offset_range = augmentation_config['systematic_offset'].get('range', 0.05)
                offset = jax.random.uniform(keys[1], minval=-offset_range, maxval=offset_range)
                data = data + offset

            # Scale variation (instrument sensitivity changes)
            if 'scale_variation' in augmentation_config:
                scale_range = augmentation_config['scale_variation'].get('range', 0.1)
                scale = jax.random.uniform(keys[2], minval=1-scale_range, maxval=1+scale_range)
                data = data * scale

            # Time series specific augmentations
            if augmentation_type == 'time_series':
                # Time warping (realistic temporal distortions)
                if 'time_warp' in augmentation_config:
                    warp_strength = augmentation_config['time_warp'].get('strength', 0.1)
                    # Simple time warping by resampling
                    indices = jnp.linspace(0, len(data)-1, len(data))
                    warp = jax.random.uniform(keys[3], indices.shape, minval=-warp_strength, maxval=warp_strength)
                    warped_indices = jnp.clip(indices + warp * len(data), 0, len(data)-1)
                    data = jnp.interp(indices, warped_indices, data)

                # Missing data simulation (sensor dropouts)
                if 'dropout' in augmentation_config:
                    dropout_prob = augmentation_config['dropout'].get('prob', 0.05)
                    mask = jax.random.uniform(keys[4], data.shape) > dropout_prob
                    data = jnp.where(mask, data, jnp.nan)

            # Spectral data augmentations
            elif augmentation_type == 'spectral':
                # Baseline drift (common in spectroscopy)
                if 'baseline_drift' in augmentation_config:
                    drift_amplitude = augmentation_config['baseline_drift'].get('amplitude', 0.02)
                    x = jnp.linspace(0, 1, len(data))
                    drift = drift_amplitude * (jax.random.uniform(keys[5]) * x +
                                             jax.random.uniform(keys[0]) * x**2)
                    data = data + drift

                # Peak shift (instrumental variations)
                if 'peak_shift' in augmentation_config:
                    shift_pixels = augmentation_config['peak_shift'].get('max_shift', 2)
                    shift = jax.random.randint(keys[1], (), -shift_pixels, shift_pixels+1)
                    data = jnp.roll(data, shift)

            return data

        def scientific_augmentation_fn(example):
            """Main scientific augmentation function"""
            if not isinstance(example, dict):
                return example

            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            keys = jax.random.split(rng_key, 10)
            key_idx = 0

            # Detect data types and apply appropriate augmentations
            for key, value in example.items():
                if isinstance(value, (np.ndarray, jnp.ndarray)) and key != 'metadata':
                    # Determine augmentation type based on data characteristics
                    if 'time' in key.lower() or 'temporal' in key.lower():
                        augmentation_type = 'time_series'
                    elif any(term in key.lower() for term in ['spectrum', 'spectral', 'frequency']):
                        augmentation_type = 'spectral'
                    elif any(term in key.lower() for term in ['image', 'spatial', 'map']):
                        augmentation_type = 'spatial'
                    else:
                        augmentation_type = 'general'

                    # Apply augmentations with probability
                    if jax.random.uniform(keys[key_idx % len(keys)]) < augmentation_config.get('prob', 0.5):
                        example[key] = scientific_array_augmentations(
                            value, keys[key_idx % len(keys)], augmentation_type
                        )
                    key_idx += 1

            # Spatial data augmentations
            if any(k in example for k in ['image', 'map', 'spatial_data']):
                spatial_key = next(k for k in ['image', 'map', 'spatial_data'] if k in example)
                spatial_data = example[spatial_key]

                if isinstance(spatial_data, (np.ndarray, jnp.ndarray)) and len(spatial_data.shape) >= 2:
                    # Rotation (preserves physical relationships)
                    if 'rotation' in augmentation_config:
                        rotation_angles = augmentation_config['rotation'].get('angles', [0, 90, 180, 270])
                        angle_idx = jax.random.randint(keys[0], (), 0, len(rotation_angles))
                        angle = rotation_angles[angle_idx]
                        if angle == 90:
                            spatial_data = jnp.rot90(spatial_data, k=1)
                        elif angle == 180:
                            spatial_data = jnp.rot90(spatial_data, k=2)
                        elif angle == 270:
                            spatial_data = jnp.rot90(spatial_data, k=3)
                        example[spatial_key] = spatial_data

                    # Flip (preserves physical symmetries)
                    if 'flip' in augmentation_config:
                        if jax.random.uniform(keys[1]) < augmentation_config['flip'].get('prob', 0.5):
                            flip_axis = augmentation_config['flip'].get('axis', 0)
                            example[spatial_key] = jnp.flip(spatial_data, axis=flip_axis)

            # Preserve scientific metadata
            if 'metadata' in example:
                if 'augmentation_applied' not in example['metadata']:
                    example['metadata']['augmentation_applied'] = []
                example['metadata']['augmentation_applied'].extend(list(augmentation_config.keys()))

            return example

        return scientific_augmentation_fn

    def _wrap_with_profiling(self, loader: grain.DataLoader) -> grain.DataLoader:
        """Add performance profiling to data loader"""

        class ProfilingDataLoader:
            def __init__(self, base_loader, metrics_dict):
                self.base_loader = base_loader
                self.metrics = metrics_dict
                self.batch_times = []
                self.throughput_history = []

            def __iter__(self):
                start_time = time.time()
                batch_count = 0

                for batch in self.base_loader:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - start_time

                    self.batch_times.append(batch_time)

                    # Calculate throughput
                    if isinstance(batch, dict) and any(isinstance(v, (np.ndarray, jnp.ndarray)) for v in batch.values()):
                        batch_size = next(v.shape[0] for v in batch.values() if isinstance(v, (np.ndarray, jnp.ndarray)))
                        throughput = batch_size / batch_time
                        self.throughput_history.append(throughput)

                    # Log metrics periodically
                    if batch_count % 100 == 0:
                        avg_batch_time = np.mean(self.batch_times[-100:])
                        avg_throughput = np.mean(self.throughput_history[-100:]) if self.throughput_history else 0

                        logging.info(f"Batch {batch_count}: "
                                   f"Avg batch time: {avg_batch_time:.3f}s, "
                                   f"Avg throughput: {avg_throughput:.1f} samples/s")

                    yield batch
                    start_time = time.time()
                    batch_count += 1

                # Store final metrics
                self.metrics.update({
                    'avg_batch_time': np.mean(self.batch_times),
                    'total_batches': batch_count,
                    'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0
                })

        return ProfilingDataLoader(loader, self.performance_metrics)

# Advanced distributed data loading
class DistributedGrainLoader:
    """Distributed data loading with sharding and coordination"""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def create_sharded_pipeline(self, data_source: str, global_batch_size: int,
                              transform_fn: Optional[Callable] = None) -> grain.DataLoader:
        """Create sharded data pipeline for distributed training"""

        # Calculate per-device batch size
        per_device_batch_size = global_batch_size // self.world_size

        # Create base dataset
        if data_source.endswith('.parquet'):
            dataset = grain.experimental.lazy_dataset.LazyMapDataset.from_parquet(data_source)
        else:
            # Generic file source
            file_list = list(Path(data_source).glob('**/*'))
            dataset = grain.MapDataset.source(file_list)

        # Apply sharding
        sharded_dataset = dataset.shard(
            num_shards=self.world_size,
            index=self.rank
        )

        # Build pipeline
        pipeline = sharded_dataset

        if transform_fn:
            pipeline = pipeline.map(transform_fn, num_parallel_calls=4)

        pipeline = (
            pipeline
            .shuffle(buffer_size=1000)
            .batch(per_device_batch_size, drop_remainder=True)
            .prefetch(2)
        )

        return grain.DataLoader(
            data_source=pipeline,
            worker_count=4,
            worker_buffer_size=2
        )

    def synchronize_epoch(self, epoch: int):
        """Synchronize epoch across all workers for deterministic shuffling"""
        # In a real distributed setup, this would use collective communication
        # For now, return a deterministic seed based on epoch and rank
        return jax.random.PRNGKey(epoch * self.world_size + self.rank)
```

### High-Performance Data Transformations

```python
# Optimized transformation library for scientific data
class ScientificDataTransformations:
    """High-performance transformations for scientific computing datasets"""

    @staticmethod
    @jax.jit
    def normalize_scientific_data(data: jnp.ndarray,
                                mean: Optional[jnp.ndarray] = None,
                                std: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """JAX-optimized scientific data normalization"""
        if mean is None:
            mean = jnp.mean(data, axis=0, keepdims=True)
        if std is None:
            std = jnp.std(data, axis=0, keepdims=True)

        return (data - mean) / (std + 1e-8)

    @staticmethod
    @jax.jit
    def spectral_preprocessing(signal: jnp.ndarray,
                             sample_rate: float = 1.0,
                             n_fft: int = 512) -> Dict[str, jnp.ndarray]:
        """Spectral analysis preprocessing for time series data"""

        # Compute FFT
        fft_result = jnp.fft.fft(signal, n=n_fft)
        magnitude = jnp.abs(fft_result)
        phase = jnp.angle(fft_result)

        # Frequency bins
        frequencies = jnp.fft.fftfreq(n_fft, d=1.0/sample_rate)

        # Power spectral density
        psd = magnitude**2 / (sample_rate * n_fft)

        return {
            'magnitude_spectrum': magnitude,
            'phase_spectrum': phase,
            'power_spectral_density': psd,
            'frequencies': frequencies
        }

    @staticmethod
    def create_sliding_window_dataset(data: jnp.ndarray,
                                    window_size: int,
                                    stride: int = 1,
                                    padding: str = 'valid') -> grain.DataSource:
        """Create sliding window dataset for sequence modeling"""

        def sliding_window_generator():
            for i in range(0, len(data) - window_size + 1, stride):
                window = data[i:i + window_size]
                yield {
                    'input': window[:-1],
                    'target': window[1:],
                    'window_start': i
                }

        return grain.MapDataset.source(sliding_window_generator())

    @staticmethod
    def create_multiresolution_dataset(images: List[jnp.ndarray],
                                     scales: List[int] = [64, 128, 256]) -> grain.DataSource:
        """Create multi-resolution image dataset"""

        def multiresolution_generator():
            for image in images:
                multi_scale = {}
                for scale in scales:
                    # Simple resize using JAX (in practice, use proper interpolation)
                    if len(image.shape) == 3:  # RGB image
                        h, w, c = image.shape
                        # Simplified resize - in practice use jax.image.resize
                        step_h = max(1, h // scale)
                        step_w = max(1, w // scale)
                        resized = image[::step_h, ::step_w, :]
                    else:  # Grayscale
                        h, w = image.shape
                        step_h = max(1, h // scale)
                        step_w = max(1, w // scale)
                        resized = image[::step_h, ::step_w]

                    multi_scale[f'scale_{scale}'] = resized

                yield multi_scale

        return grain.MapDataset.source(multiresolution_generator())

    @staticmethod
    def create_augmented_dataset(base_dataset: grain.DataSource,
                               augmentation_factor: int = 5) -> grain.DataSource:
        """Create augmented dataset with multiple versions of each sample"""

        def augmentation_generator():
            for original_sample in base_dataset:
                # Yield original sample
                yield {**original_sample, 'augmentation_id': 0}

                # Yield augmented versions
                for aug_id in range(1, augmentation_factor):
                    rng_key = jax.random.PRNGKey(aug_id)
                    augmented_sample = ScientificDataTransformations._apply_random_augmentation(
                        original_sample, rng_key
                    )
                    yield {**augmented_sample, 'augmentation_id': aug_id}

        return grain.MapDataset.source(augmentation_generator())

    @staticmethod
    @jax.jit
    def _apply_random_augmentation(sample: Dict, rng_key: jax.random.PRNGKey) -> Dict:
        """Apply random augmentation to sample"""
        keys = jax.random.split(rng_key, 3)

        augmented = sample.copy()

        # Example augmentations for different data types
        if 'image' in sample:
            image = sample['image']

            # Random noise
            noise_level = jax.random.uniform(keys[0], minval=0.0, maxval=0.05)
            noise = jax.random.normal(keys[1], image.shape) * noise_level
            augmented['image'] = jnp.clip(image + noise, 0.0, 1.0)

        if 'signal' in sample:
            signal = sample['signal']

            # Random time shift
            shift = jax.random.randint(keys[2], (), 0, len(signal) // 10)
            augmented['signal'] = jnp.roll(signal, shift)

        return augmented

# Memory-efficient data loading for large datasets
class MemoryEfficientLoader:
    """Memory-efficient loading strategies for large scientific datasets"""

    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3

    def create_streaming_loader(self, file_pattern: str,
                              chunk_size: int = 1000) -> Iterator[Dict]:
        """Create streaming loader for large files"""

        file_paths = list(Path('.').glob(file_pattern))

        for file_path in file_paths:
            if file_path.suffix == '.h5':
                yield from self._stream_hdf5(file_path, chunk_size)
            elif file_path.suffix == '.zarr':
                yield from self._stream_zarr(file_path, chunk_size)
            elif file_path.suffix == '.parquet':
                yield from self._stream_parquet(file_path, chunk_size)

    def _stream_hdf5(self, file_path: Path, chunk_size: int) -> Iterator[Dict]:
        """Stream HDF5 file in chunks"""
        import h5py

        with h5py.File(file_path, 'r') as f:
            dataset_names = list(f.keys())

            # Determine dataset length
            first_dataset = f[dataset_names[0]]
            total_length = len(first_dataset)

            # Stream in chunks
            for start_idx in range(0, total_length, chunk_size):
                end_idx = min(start_idx + chunk_size, total_length)

                chunk_data = {}
                for dataset_name in dataset_names:
                    chunk_data[dataset_name] = f[dataset_name][start_idx:end_idx]

                # Yield individual samples from chunk
                for i in range(len(chunk_data[dataset_names[0]])):
                    sample = {name: data[i] for name, data in chunk_data.items()}
                    yield sample

    def _stream_zarr(self, file_path: Path, chunk_size: int) -> Iterator[Dict]:
        """Stream Zarr array in chunks"""
        import zarr

        store = zarr.open(file_path, mode='r')

        if isinstance(store, zarr.Group):
            array_names = list(store.array_keys())
            total_length = len(store[array_names[0]])

            for start_idx in range(0, total_length, chunk_size):
                end_idx = min(start_idx + chunk_size, total_length)

                chunk_data = {}
                for array_name in array_names:
                    chunk_data[array_name] = store[array_name][start_idx:end_idx]

                for i in range(len(chunk_data[array_names[0]])):
                    sample = {name: data[i] for name, data in chunk_data.items()}
                    yield sample

    def _stream_parquet(self, file_path: Path, chunk_size: int) -> Iterator[Dict]:
        """Stream Parquet file in chunks"""
        import pandas as pd

        # Read in chunks
        for chunk_df in pd.read_parquet(file_path, chunksize=chunk_size):
            for _, row in chunk_df.iterrows():
                yield row.to_dict()

    def estimate_memory_usage(self, sample_data: Dict) -> int:
        """Estimate memory usage of a data sample"""
        total_bytes = 0

        for key, value in sample_data.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                total_bytes += value.nbytes
            elif isinstance(value, str):
                total_bytes += len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                total_bytes += 8  # Approximate
            elif isinstance(value, list):
                total_bytes += len(value) * 8  # Approximate

        return total_bytes

    def create_memory_aware_batching(self, dataset: grain.DataSource,
                                   target_memory_usage: Optional[int] = None) -> grain.DataSource:
        """Create batches based on memory usage rather than fixed size"""

        if target_memory_usage is None:
            target_memory_usage = self.max_memory_bytes // 10  # 10% of available memory

        def memory_aware_batch_generator():
            current_batch = []
            current_memory = 0

            for sample in dataset:
                sample_memory = self.estimate_memory_usage(sample)

                if current_memory + sample_memory > target_memory_usage and current_batch:
                    # Yield current batch and start new one
                    yield self._stack_batch(current_batch)
                    current_batch = [sample]
                    current_memory = sample_memory
                else:
                    current_batch.append(sample)
                    current_memory += sample_memory

            # Yield final batch
            if current_batch:
                yield self._stack_batch(current_batch)

        return grain.MapDataset.source(memory_aware_batch_generator())

    def _stack_batch(self, samples: List[Dict]) -> Dict:
        """Stack list of samples into batched arrays"""
        if not samples:
            return {}

        # Get keys from first sample
        keys = samples[0].keys()
        batched = {}

        for key in keys:
            values = [sample[key] for sample in samples]

            # Stack arrays
            if all(isinstance(v, (np.ndarray, jnp.ndarray)) for v in values):
                batched[key] = jnp.stack(values)
            # Handle lists of same type
            elif all(isinstance(v, (int, float)) for v in values):
                batched[key] = jnp.array(values)
            # Keep as list for mixed types
            else:
                batched[key] = values

        return batched
```

### Cross-Framework Integration

```python
# Integration with different ML frameworks
class CrossFrameworkDataAdapter:
    """Adapt Grain datasets for different ML frameworks"""

    @staticmethod
    def to_tensorflow_dataset(grain_loader: grain.DataLoader) -> 'tf.data.Dataset':
        """Convert Grain loader to TensorFlow Dataset"""
        import tensorflow as tf

        def grain_generator():
            for batch in grain_loader:
                # Convert JAX arrays to TensorFlow tensors
                tf_batch = {}
                for key, value in batch.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        tf_batch[key] = tf.convert_to_tensor(np.array(value))
                    else:
                        tf_batch[key] = value
                yield tf_batch

        # Determine output signature
        sample_batch = next(iter(grain_loader))
        output_signature = {}
        for key, value in sample_batch.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                output_signature[key] = tf.TensorSpec(
                    shape=value.shape,
                    dtype=tf.as_dtype(value.dtype)
                )

        return tf.data.Dataset.from_generator(
            grain_generator,
            output_signature=output_signature
        )

    @staticmethod
    def to_pytorch_dataloader(grain_loader: grain.DataLoader,
                            num_workers: int = 4) -> 'torch.utils.data.DataLoader':
        """Convert Grain loader to PyTorch DataLoader"""
        import torch
        from torch.utils.data import Dataset, DataLoader

        class GrainDataset(Dataset):
            def __init__(self, grain_loader):
                # Pre-load all data (for finite datasets)
                self.data = list(grain_loader)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                batch = self.data[idx]
                # Convert to PyTorch tensors
                torch_batch = {}
                for key, value in batch.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        torch_batch[key] = torch.from_numpy(np.array(value))
                    else:
                        torch_batch[key] = value
                return torch_batch

        dataset = GrainDataset(grain_loader)
        return DataLoader(
            dataset,
            batch_size=1,  # Already batched by Grain
            num_workers=num_workers,
            pin_memory=True
        )

    @staticmethod
    def create_jax_data_iterator(grain_loader: grain.DataLoader) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create JAX-optimized data iterator"""

        def jax_iterator():
            for batch in grain_loader:
                # Ensure all arrays are JAX arrays
                jax_batch = {}
                for key, value in batch.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        # Move to device if specified
                        jax_batch[key] = jax.device_put(jnp.array(value))
                    else:
                        jax_batch[key] = value
                yield jax_batch

        return jax_iterator()

    @staticmethod
    def create_flax_data_iterator(grain_loader: grain.DataLoader,
                                 rng_key: jax.random.PRNGKey) -> Iterator[Dict]:
        """Create Flax-compatible data iterator with PRNG handling"""

        def flax_iterator():
            nonlocal rng_key

            for batch in grain_loader:
                # Split RNG key for this batch
                rng_key, batch_key = jax.random.split(rng_key)

                # Add RNG key to batch for stochastic operations
                flax_batch = {
                    'data': batch,
                    'rng_key': batch_key
                }

                # Ensure data is on correct device
                for key, value in flax_batch['data'].items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        flax_batch['data'][key] = jax.device_put(jnp.array(value))

                yield flax_batch

        return flax_iterator()

# Performance monitoring and optimization
class DataLoadingProfiler:
    """Comprehensive data loading performance profiler"""

    def __init__(self):
        self.metrics = {
            'throughput': [],
            'latency': [],
            'memory_usage': [],
            'cpu_usage': [],
            'io_wait': []
        }

    def profile_pipeline(self, data_loader: grain.DataLoader,
                        num_batches: int = 100) -> Dict:
        """Profile data loading pipeline performance"""
        import psutil
        import time

        start_time = time.time()
        batch_times = []
        memory_usage = []

        for i, batch in enumerate(data_loader):
            batch_start = time.time()

            # Force computation to measure actual data loading time
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        _ = jnp.sum(value)  # Force computation

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

            # Memory usage
            memory_usage.append(psutil.virtual_memory().percent)

            if i >= num_batches:
                break

        total_time = time.time() - start_time

        return {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times),
            'throughput_batches_per_sec': len(batch_times) / total_time,
            'avg_memory_usage': np.mean(memory_usage),
            'max_memory_usage': np.max(memory_usage),
            'batch_time_percentiles': {
                '50%': np.percentile(batch_times, 50),
                '90%': np.percentile(batch_times, 90),
                '95%': np.percentile(batch_times, 95),
                '99%': np.percentile(batch_times, 99)
            }
        }

    def optimize_pipeline(self, pipeline_config: Dict) -> Dict:
        """Suggest optimizations based on profiling results"""
        optimizations = []

        # Check throughput
        if self.metrics.get('throughput_batches_per_sec', 0) < 10:
            optimizations.append({
                'issue': 'Low throughput',
                'suggestion': 'Increase worker_count or enable prefetching',
                'config_change': {
                    'worker_count': min(pipeline_config.get('worker_count', 1) * 2, 8),
                    'prefetch_buffer_size': max(pipeline_config.get('prefetch_buffer_size', 1), 4)
                }
            })

        # Check memory usage
        avg_memory = self.metrics.get('avg_memory_usage', 0)
        if avg_memory > 80:
            optimizations.append({
                'issue': 'High memory usage',
                'suggestion': 'Reduce batch size or enable streaming',
                'config_change': {
                    'batch_size': max(pipeline_config.get('batch_size', 32) // 2, 1),
                    'worker_buffer_size': min(pipeline_config.get('worker_buffer_size', 2), 1)
                }
            })

        # Check latency variation
        batch_times = self.metrics.get('batch_times', [])
        if batch_times and np.std(batch_times) / np.mean(batch_times) > 0.5:
            optimizations.append({
                'issue': 'High latency variation',
                'suggestion': 'Enable deterministic processing and increase buffer sizes',
                'config_change': {
                    'shuffle_buffer_size': max(pipeline_config.get('shuffle_buffer_size', 1000), 2000),
                    'deterministic': True
                }
            })

        return {
            'optimizations': optimizations,
            'performance_score': self._calculate_performance_score(),
            'bottleneck_analysis': self._identify_bottlenecks()
        }

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100.0

        # Penalize low throughput
        throughput = self.metrics.get('throughput_batches_per_sec', 0)
        if throughput < 5:
            score -= 30
        elif throughput < 10:
            score -= 15

        # Penalize high memory usage
        memory_usage = self.metrics.get('avg_memory_usage', 0)
        if memory_usage > 90:
            score -= 25
        elif memory_usage > 80:
            score -= 10

        # Penalize high latency variation
        batch_times = self.metrics.get('batch_times', [1])
        if len(batch_times) > 1:
            cv = np.std(batch_times) / np.mean(batch_times)
            if cv > 1.0:
                score -= 20
            elif cv > 0.5:
                score -= 10

        return max(0.0, score)

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        throughput = self.metrics.get('throughput_batches_per_sec', 0)
        memory_usage = self.metrics.get('avg_memory_usage', 0)

        if throughput < 5:
            bottlenecks.append('CPU-bound: Consider increasing parallelism')

        if memory_usage > 80:
            bottlenecks.append('Memory-bound: Reduce batch size or enable streaming')

        # Check I/O patterns
        batch_times = self.metrics.get('batch_times', [])
        if batch_times and np.std(batch_times) / np.mean(batch_times) > 0.5:
            bottlenecks.append('I/O-bound: Consider prefetching or SSD storage')

        return bottlenecks
```

## Use Cases and Scientific Applications

### Computational Biology and Genomics
- **Large-Scale Genomic Data**: Efficient loading of VCF, FASTA, and BAM files with parallel processing
- **Single-Cell Analysis**: Memory-efficient streaming of high-dimensional single-cell datasets
- **Phylogenetic Data**: Multi-sequence alignment data with sliding window preprocessing
- **Protein Structure**: PDB file processing with 3D structure augmentations

### Climate Science and Earth Observation
- **Satellite Imagery**: Multi-temporal, multi-spectral remote sensing data pipelines
- **Climate Model Output**: NetCDF and HDF5 ensemble data with temporal chunking
- **Weather Station Data**: Time series preprocessing with quality filtering and gap filling
- **Atmospheric Data**: Vertical profile data with interpolation and resampling

### Physics and Materials Science
- **Simulation Data**: Molecular dynamics trajectories with sliding window analysis
- **Experimental Data**: High-throughput experimental datasets with real-time processing
- **Spectroscopy Data**: Multi-dimensional spectral data with preprocessing and normalization
- **Crystallographic Data**: Structure data with symmetry operations and augmentations

### Neuroscience and Medical Imaging
- **fMRI Data**: 4D brain imaging with temporal and spatial preprocessing
- **EEG/MEG Signals**: High-frequency neural signals with artifact removal
- **Medical Images**: DICOM processing with privacy-preserving transformations
- **Behavioral Data**: Multi-modal behavioral datasets with synchronization

## Integration with Existing Agents

- **JAX Expert**: Advanced JAX transformations for data preprocessing and augmentation
- **GPU Computing Expert**: Memory optimization and distributed data loading strategies
- **ML Engineer**: Production data pipeline deployment and monitoring
- **Statistics Expert**: Statistical data quality assessment and preprocessing validation
- **Visualization Expert**: Data pipeline monitoring and performance visualization
- **Experiment Manager**: Systematic data pipeline experimentation and optimization

## Practical Usage Examples

### Example 1: Climate Science Multi-Dataset Analysis
```python
# Loading climate model ensemble data with observational validation
loader = ScientificDataLoader(worker_count=8, memory_limit_gb=16.0)

data_sources = {
    'model_output': '/data/climate/model_ensemble.h5',
    'observations': '/data/climate/station_data.npz',
    'reanalysis': '/data/climate/era5_data.zarr'
}

scientific_config = {
    'hdf5_config': {
        'datasets': ['temperature', 'precipitation', 'pressure'],
        'chunk_axis': 0  # Time dimension
    },
    'quality_thresholds': {
        'temperature': (-100, 60),  # Valid temperature range in Celsius
        'precipitation': (0, 1000),  # Valid precipitation range in mm
        'min_variance': 1e-6
    },
    'preprocessing': {
        'normalize': {'mean': [15.0, 2.0, 1013.25], 'std': [20.0, 5.0, 30.0]},
        'temporal_interpolation': True
    },
    'augmentations': {
        'gaussian_noise': {'std': 0.02},
        'temporal_shift': {'max_hours': 6},
        'prob': 0.3
    }
}

climate_pipeline = loader.create_scientific_pipeline(
    data_sources, scientific_config, batch_size=64, deterministic=True
)

# Process temporal sequences for climate trend analysis
for batch in climate_pipeline:
    # batch contains aligned multi-source climate data
    model_data = batch['datasets']['model_output']
    obs_data = batch['datasets']['observations']
    # Perform climate analysis...
```

### Example 2: Genomics Multi-Omics Data Integration
```python
# Integrating genomics, transcriptomics, and proteomics data
genomics_loader = ScientificDataLoader(worker_count=6, chunk_size=5000)

omics_sources = {
    'genotype': '/data/genomics/variants.npz',
    'expression': '/data/genomics/rna_seq.h5',
    'methylation': '/data/genomics/methylation_array.csv',
    'phenotype': '/data/genomics/clinical_data.xlsx'
}

omics_config = {
    'hdf5_config': {
        'datasets': ['gene_expression', 'sample_metadata'],
        'chunk_axis': 1  # Sample dimension
    },
    'csv_config': {
        'numeric_columns': ['beta_value', 'p_value'],
        'chunk_size': 10000
    },
    'excel_config': {
        'sheet_name': 'clinical_data',
        'header_row': 0
    },
    'quality_thresholds': {
        'gene_expression': (0, 20),  # Log2 expression values
        'beta_value': (0, 1),  # Methylation beta values
        'required_metadata': ['sample_id', 'tissue_type']
    },
    'preprocessing': {
        'log_transform': True,
        'batch_correction': True,
        'missing_value_imputation': 'mean'
    }
}

omics_pipeline = genomics_loader.create_scientific_pipeline(
    omics_sources, omics_config, batch_size=32
)
```

### Example 3: Physics Simulation Data Processing
```python
# Processing molecular dynamics simulation trajectories
physics_loader = ScientificDataLoader(worker_count=12, memory_limit_gb=32.0)

simulation_sources = {
    'trajectory': '/data/md/trajectory.h5',
    'forces': '/data/md/forces.npz',
    'energies': '/data/md/energies.csv'
}

physics_config = {
    'hdf5_config': {
        'datasets': ['coordinates', 'velocities', 'box_vectors'],
        'chunk_axis': 0  # Time frames
    },
    'quality_thresholds': {
        'coordinates': (-100, 100),  # Coordinate bounds in Angstroms
        'energies': (-1e6, 1e6),  # Energy bounds in kJ/mol
        'min_array_size': 100  # Minimum trajectory length
    },
    'preprocessing': {
        'center_coordinates': True,
        'remove_drift': True,
        'calculate_distances': True
    },
    'augmentations': {
        'coordinate_noise': {'std': 0.01},  # Thermal noise simulation
        'systematic_offset': {'range': 0.001},
        'prob': 0.4
    }
}

md_pipeline = physics_loader.create_scientific_pipeline(
    simulation_sources, physics_config, batch_size=128
)

# Process simulation frames for analysis
for batch in md_pipeline:
    coords = batch['datasets']['trajectory']['coordinates']
    forces = batch['datasets']['forces']['data']
    # Perform molecular analysis...
```

### Example 4: Medical Imaging Multi-Modal Pipeline
```python
# Processing multi-modal medical imaging data
medical_loader = ScientificDataLoader(worker_count=4, memory_limit_gb=24.0)

imaging_sources = {
    'mri_t1': '/data/medical/t1_weighted/',  # Directory of DICOM files
    'mri_t2': '/data/medical/t2_weighted.h5',
    'clinical': '/data/medical/patient_data.xlsx'
}

medical_config = {
    'quality_thresholds': {
        'intensity': (0, 4095),  # Valid MRI intensity range
        'min_variance': 1e-4,  # Avoid blank images
        'required_metadata': ['patient_id', 'acquisition_date']
    },
    'preprocessing': {
        'skull_stripping': True,
        'intensity_normalization': True,
        'spatial_registration': True
    },
    'augmentations': {
        'gaussian_noise': {'std': 0.01},
        'intensity_scaling': {'range': 0.1},
        'spatial_rotation': {'max_degrees': 5},
        'prob': 0.5
    }
}

medical_pipeline = medical_loader.create_scientific_pipeline(
    imaging_sources, medical_config, batch_size=8
)
```

## Advanced Scientific Data Pipeline Patterns

### Temporal Alignment for Multi-Instrument Data
```python
# Synchronize data from multiple scientific instruments
def create_temporal_sync_pipeline(instrument_data_paths, time_tolerance=1.0):
    \"\"\"Create pipeline for temporally synchronized multi-instrument data\"\"\"

    config = {
        'quality_thresholds': {
            'time_gap_seconds': time_tolerance,
            'required_metadata': ['timestamp', 'instrument_id']
        },
        'preprocessing': {
            'temporal_interpolation': {
                'method': 'linear',
                'max_gap_seconds': time_tolerance * 2
            },
            'synchronization_window': time_tolerance
        }
    }

    return loader.create_scientific_pipeline(
        instrument_data_paths, config, deterministic=True
    )
```

### Spatial Data Fusion for Earth Observation
```python
# Combine satellite data with ground measurements
def create_spatial_fusion_pipeline(satellite_data, ground_stations):
    \"\"\"Create pipeline for spatial data fusion\"\"\"

    config = {
        'quality_thresholds': {
            'spatial_resolution_m': (10, 1000),  # Valid resolution range
            'cloud_cover_percent': (0, 20),  # Max cloud cover
            'required_metadata': ['coordinates', 'acquisition_time']
        },
        'preprocessing': {
            'spatial_interpolation': {
                'method': 'kriging',
                'grid_resolution': 100  # meters
            },
            'atmospheric_correction': True
        }
    }

    return loader.create_scientific_pipeline(
        {'satellite': satellite_data, 'ground': ground_stations},
        config, batch_size=16
    )
```

This agent transforms traditional data loading from ad-hoc scripts into **high-performance, scalable, and maintainable data infrastructure** using Grain and modern data loading best practices, enabling efficient scientific computing workflows across diverse domains and scales.