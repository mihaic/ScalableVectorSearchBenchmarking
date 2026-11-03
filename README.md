# Scalable Vector Search Benchmarking
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/ScalableVectorSearchBenchmarking/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/ScalableVectorSearchBenchmarking)
![GitHub License](https://img.shields.io/github/license/IntelLabs/ScalableVectorSearchBenchmarking)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)

Scalable Vector Search Benchmarking enables the benchmarking or evaluation of the [Scalable Vector Search](https://github.com/intel/ScalableVectorSearch) library.

## Installation

Requires Python >= 3.12.

```sh
python -m pip install \
    "svsbench[svs] @ git+https://github.com/IntelLabs/ScalableVectorSearchBenchmarking"
```

The `svs` extra installs the SVS Python bindings, which all commands need
except the [C API commands](#using-the-svs-c-api).

## Usage

### Building an index

```sh
python -m svsbench.build \
    --vecs_file /path/to/vectors.fvecs \
    --svs_type leanvec4x8 --leanvec_dims -4 \
    --proportion_vectors_init 0.5 --batch_size 10000
```

#### Building a LeanVec-OOD index

[LeanVec-OOD](https://openreview.net/forum?id=wczqrpOrIc) is a dimensionality reduction method for cases where query and base vectors have different distributions (e.g., text-to-image search). Use `--train_query_file` to provide training queries.

```sh
python -m svsbench.build \
    --vecs_file /path/to/vectors.fvecs \
    --svs_type leanvec4x8 \
    --train_query_file /path/to/train_query_vectors.fvecs
```

### Computing the ground truth

For the query vectors used in performance measurements:
```sh
python -m svsbench.generate_ground_truth \
    --vecs_file vectors.fvecs \
    --query_file query_vectors.fvecs
```

For the query vectors used in the calibration of search parameters:
```sh
python -m svsbench.generate_ground_truth \
    --vecs_file vectors.fvecs \
    --query_file calibration_query_vectors.fvecs
```

### Searching

Calibrating the search parameters for a given recall and then searching:
```sh
python -m svsbench.search \
    --idx_dir /path/to/index_dir_from_build \
    --svs_type leanvec4x8 \
    --query_file /path/to/query_vectors.fvecs \
    --ground_truth_file /path/to/ground_truth.ivecs \
    -k 5 \
    --recall 0.95 \
    --calibration_query_file /path/to/calibration_query_vectors.fvecs \
    --calibration_ground_truth_file /path/to/calibration_ground_truth.ivecs
```

Searching using specified search parameters:
```sh
python -m svsbench.search \
    --idx_dir /path/to/index_dir_from_build \
    --svs_type leanvec4x8 \
    --query_file /path/to/query_vectors.fvecs \
    --ground_truth_file /path/to/ground_truth.ivecs \
    -k 5 \
    --batch_size 1 \
    --search_window_size 14 \
    --search_buffer_capacity 34 \
    --prefetch_lookahead 1 \
    --prefetch_step 0 \
    --batch_size 10000 \
    --search_window_size 15 \
    --search_buffer_capacity 36 \
    --prefetch_lookahead 10 \
    --prefetch_step 4
```

## Using the SVS C API

`svsbench.build_capi` and `svsbench.search_capi` benchmark a dynamic index
through the [SVS C API](https://github.com/intel/ScalableVectorSearch/tree/main/bindings/c)
instead of the Python bindings. The index uses `SVS_SYNC_KIND_FINE_GRAIN`:
`--max_threads` threads call the C API concurrently, each call adding or
searching `--call_size` vectors (default 1, the maximum supported by the
concurrent index). `--pool_threads` sets the index's own thread pool. With
`SVS_SYNC_KIND_FINE_GRAIN`, the work of each call runs on that pool, so
`build_capi` defaults it to `--max_threads`. `search_capi` defaults it to 1,
which runs each call on its calling thread instead. `--threadpool omp` uses an
OpenMP pool if the library was built with OpenMP.

These commands do not need the SVS Python bindings, so they run on
free-threaded Python, where the calling threads run in parallel:

```sh
uv venv --python 3.14t
uv pip install git+https://github.com/IntelLabs/ScalableVectorSearchBenchmarking
```

Set `SVS_C_API_LIB` to the path of `libsvs_c_api.so`, e.g., from an
[SVS release](https://github.com/intel/ScalableVectorSearch/releases):

```sh
export SVS_C_API_LIB=/path/to/svs-c-api/lib/libsvs_c_api.so
```

Building an index:

```sh
python -m svsbench.build_capi \
    --vecs_file /path/to/vectors.fvecs \
    --svs_type leanvec4x8 --leanvec_dims -4 \
    --proportion_vectors_init 0.5 --batch_size 10000 \
    --max_threads 32
```

Searching with specified search window sizes (the C API does not support
calibration):

```sh
python -m svsbench.search_capi \
    --idx_dir /path/to/index_dir_from_build_capi \
    --svs_type leanvec4x8 \
    --query_file /path/to/query_vectors.fvecs \
    --ground_truth_file /path/to/ground_truth.ivecs \
    -k 10 \
    --search_window_size 20 --search_window_size 40 \
    --max_threads 32
```

The search logs the throughput and the latency per C API call. The C API
supports fewer build parameters than the Python bindings: `prune_to` is
`graph_max_degree - 4` and `max_candidate_pool_size` is `3 * window_size`.
