# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Thin ctypes wrapper of the SVS C API for dynamic Vamana indices.

The shared library is found through the SVS_C_API_LIB environment
variable (path to libsvs_c_api.so) or, failing that, the system library
search path. The wrapper does not import the SVS Python bindings, so it
works on free-threaded Python.

Error handles are per thread, as required by the C API for indices
with internal synchronization.
"""

import ctypes
import ctypes.util
import enum
import functools
import os
import threading
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

LIB_ENV: Final = "SVS_C_API_LIB"

# Version of svs_c.h whose struct layouts are mirrored below.
_HEADER_VERSION: Final = (0 << 16) | (5 << 8) | 0


class ErrorCode(enum.IntEnum):
    OK = 0
    GENERIC = 1
    INVALID_ARGUMENT = 2
    OUT_OF_MEMORY = 3
    NOT_IMPLEMENTED = 5
    UNSUPPORTED_HW = 6
    RUNTIME = 7
    INVALID_OPERATION = 8
    UNKNOWN = 1000


THREADPOOLS: Final = ("native", "omp")


class SyncKind(enum.IntEnum):
    NONE = 0
    GLOBAL = 1
    FINE_GRAIN = 2


class _DistanceMetric(enum.IntEnum):
    EUCLIDEAN = 0
    COSINE = 1
    DOT_PRODUCT = 2


class _DataType(enum.IntEnum):
    NONE = 0
    UINT4 = (1 << 8) | 4
    UINT8 = (1 << 8) | 8
    FLOAT16 = (3 << 8) | 16
    FLOAT32 = (3 << 8) | 32


class _ThreadpoolKind(enum.IntEnum):
    NATIVE = 0
    OMP = 1
    SINGLE_THREAD = 2


STR_TO_METRIC: Final = {
    "cosine": _DistanceMetric.COSINE,
    "l2": _DistanceMetric.EUCLIDEAN,
    "mip": _DistanceMetric.DOT_PRODUCT,
}

_LVQ_TYPES: Final = {
    "lvq8": (_DataType.UINT8, _DataType.NONE),
    "lvq4x4": (_DataType.UINT4, _DataType.UINT4),
    "lvq4x8": (_DataType.UINT4, _DataType.UINT8),
}

_LEANVEC_TYPES: Final = {
    "leanvec4x4": (_DataType.UINT4, _DataType.UINT4),
    "leanvec4x8": (_DataType.UINT4, _DataType.UINT8),
    "leanvec8x8": (_DataType.UINT8, _DataType.UINT8),
}

_SIMPLE_TYPES: Final = {
    "float16": _DataType.FLOAT16,
    "float32": _DataType.FLOAT32,
}


class _DynamicIndexParams(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("blocksize_bytes", ctypes.c_size_t),
        ("blocksize_elements", ctypes.c_size_t),
        ("sync_kind", ctypes.c_uint32),
    )


class _SearchResults(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("num_queries", ctypes.c_size_t),
        ("total_results", ctypes.c_size_t),
        ("offsets", ctypes.c_void_p),
        ("indices", ctypes.c_void_p),
        ("distances", ctypes.c_void_p),
        ("offsets_capacity", ctypes.c_size_t),
        ("results_capacity", ctypes.c_size_t),
        ("owns_buffers", ctypes.c_bool),
    )


_H = ctypes.c_void_p
_SIZE = ctypes.c_size_t
_SIZE_P = ctypes.POINTER(ctypes.c_size_t)
_BOOL = ctypes.c_bool
_INT = ctypes.c_int

_PROTOTYPES: Final = {
    "svs_get_version_string": (ctypes.c_char_p, ()),
    "svs_error_create": (_H, ()),
    "svs_error_get_code": (_INT, (_H,)),
    "svs_error_get_message": (ctypes.c_char_p, (_H,)),
    "svs_error_free": (None, (_H,)),
    "svs_algorithm_create_vamana": (_H, (_SIZE, _SIZE, _SIZE, _H)),
    "svs_algorithm_vamana_set_alpha": (_BOOL, (_H, ctypes.c_float, _H)),
    "svs_algorithm_free": (None, (_H,)),
    "svs_storage_create_simple": (_H, (_INT, _H)),
    "svs_storage_create_lvq": (_H, (_INT, _INT, _H)),
    "svs_storage_create_leanvec": (_H, (_SIZE, _INT, _INT, _H)),
    "svs_storage_create_leanvec_trained": (_H, (_H, _INT, _INT, _H)),
    "svs_storage_free": (None, (_H,)),
    "svs_leanvec_training_data_build": (
        _H,
        (_H, _SIZE, _SIZE, ctypes.c_void_p, _SIZE, ctypes.c_void_p, _H),
    ),
    "svs_leanvec_training_data_free": (None, (_H,)),
    "svs_index_builder_create": (_H, (_INT, _SIZE, _H, _H)),
    "svs_index_builder_set_storage": (_BOOL, (_H, _H, _H)),
    "svs_index_builder_set_threadpool": (_BOOL, (_H, _INT, _SIZE, _H)),
    "svs_index_builder_free": (None, (_H,)),
    "svs_index_build_dynamic_ex": (
        _H,
        (
            _H,
            ctypes.c_void_p,
            ctypes.c_void_p,
            _SIZE,
            ctypes.POINTER(_DynamicIndexParams),
            _H,
        ),
    ),
    "svs_index_convert_dynamic_ex": (
        _H,
        (_H, _H, ctypes.POINTER(_DynamicIndexParams), _H),
    ),
    "svs_index_load_dynamic_ex": (
        _H,
        (_H, ctypes.c_char_p, ctypes.POINTER(_DynamicIndexParams), _H),
    ),
    "svs_index_free": (None, (_H,)),
    "svs_index_get_size": (_BOOL, (_H, _SIZE_P, _H)),
    "svs_index_save": (_BOOL, (_H, ctypes.c_char_p, _H)),
    "svs_index_set_num_threads": (_BOOL, (_H, _SIZE, _H)),
    "svs_index_get_num_threads": (_BOOL, (_H, _SIZE_P, _H)),
    "svs_index_dynamic_get_sync_kind": (
        _BOOL,
        (_H, ctypes.POINTER(ctypes.c_uint32), _H),
    ),
    "svs_index_dynamic_add_points": (
        _BOOL,
        (_H, ctypes.c_void_p, ctypes.c_void_p, _SIZE, _SIZE_P, _H),
    ),
    "svs_index_dynamic_delete_points": (
        _BOOL,
        (_H, ctypes.c_void_p, _SIZE, _SIZE_P, _H),
    ),
    "svs_search_params_create_vamana": (_H, (_SIZE, _H)),
    "svs_search_params_free": (None, (_H,)),
    "svs_index_search_topk": (
        _BOOL,
        (
            _H,
            ctypes.c_void_p,
            _SIZE,
            _SIZE,
            ctypes.POINTER(_SearchResults),
            _H,
            _H,
            _H,
        ),
    ),
}


class CAPIError(RuntimeError):
    """Error reported by the SVS C API."""

    def __init__(self, function: str, code: int, message: str) -> None:
        super().__init__(f"{function} failed with code {code}: {message}")
        self.code = code


@functools.cache
def lib() -> ctypes.CDLL:
    """Load the SVS C API shared library."""
    path = os.environ.get(LIB_ENV) or ctypes.util.find_library("svs_c_api")
    if path is None:
        raise OSError(
            f"SVS C API library not found. Set {LIB_ENV} to the path"
            " of libsvs_c_api.so."
        )
    cdll = ctypes.CDLL(path)
    for name, (restype, argtypes) in _PROTOTYPES.items():
        function = getattr(cdll, name)
        function.restype = restype
        function.argtypes = argtypes
    return cdll


def library_path() -> str | None:
    """Return the path of the SVS C API library or None if unknown."""
    path = os.environ.get(LIB_ENV) or ctypes.util.find_library("svs_c_api")
    return str(Path(path).resolve()) if path and Path(path).exists() else path


def version() -> str:
    """Return the version string of the SVS C API library."""
    return lib().svs_get_version_string().decode()


class _ThreadError(threading.local):
    # Each handle lives as long as the process; there is one per thread.
    def __init__(self) -> None:
        self.handle = lib().svs_error_create()


_thread_error = None
_thread_error_lock = threading.Lock()


def _error() -> int:
    """Return the error handle of the current thread."""
    global _thread_error
    if _thread_error is None:
        with _thread_error_lock:
            if _thread_error is None:
                _thread_error = _ThreadError()
    return _thread_error.handle


def _raise(function: str, err: int) -> None:
    c_lib = lib()
    message = c_lib.svs_error_get_message(err)
    raise CAPIError(
        function,
        c_lib.svs_error_get_code(err),
        message.decode() if message else "",
    )


def _call(function: str, *args):
    """Call a C API function returning a handle or bool and check errors."""
    err = _error()
    result = getattr(lib(), function)(*args, err)
    if not result:
        _raise(function, err)
    return result


def _as_float32(array: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_ids(array: npt.ArrayLike) -> npt.NDArray[np.uintp]:
    return np.ascontiguousarray(array, dtype=np.uintp)


def effective_leanvec_dims(leanvec_dims: int, dim: int) -> int:
    """Resolve negative LeanVec dims as dim // -leanvec_dims."""
    return dim // -leanvec_dims if leanvec_dims < 0 else leanvec_dims


class _Builder:
    """Index builder with algorithm and storage configuration."""

    def __init__(
        self,
        *,
        svs_type: str,
        distance: str,
        dim: int,
        graph_max_degree: int,
        window_size: int,
        alpha: float | None,
        num_threads: int,
        threadpool: str,
        leanvec_dims: int,
        train_vectors: npt.NDArray | None = None,
        train_queries: npt.NDArray | None = None,
    ) -> None:
        self._handles: list[tuple[str, int]] = []
        try:
            algorithm = _call(
                "svs_algorithm_create_vamana",
                graph_max_degree,
                window_size,
                window_size,
            )
            self._handles.append(("svs_algorithm_free", algorithm))
            if alpha is not None:
                _call("svs_algorithm_vamana_set_alpha", algorithm, alpha)
            self.handle = _call(
                "svs_index_builder_create",
                STR_TO_METRIC[distance],
                dim,
                algorithm,
            )
            self._handles.append(("svs_index_builder_free", self.handle))
            self.set_threadpool(num_threads, threadpool)
            storage = self._create_storage(
                svs_type, dim, leanvec_dims, train_vectors, train_queries
            )
            if storage is not None:
                self._handles.append(("svs_storage_free", storage))
                _call("svs_index_builder_set_storage", self.handle, storage)
        except BaseException:
            self.free()
            raise

    def _create_storage(
        self,
        svs_type: str,
        dim: int,
        leanvec_dims: int,
        train_vectors: npt.NDArray | None,
        train_queries: npt.NDArray | None,
    ) -> int | None:
        if svs_type == "float32":
            return None
        if svs_type in _SIMPLE_TYPES:
            return _call("svs_storage_create_simple", _SIMPLE_TYPES[svs_type])
        if svs_type in _LVQ_TYPES:
            return _call("svs_storage_create_lvq", *_LVQ_TYPES[svs_type])
        if svs_type in _LEANVEC_TYPES:
            primary, secondary = _LEANVEC_TYPES[svs_type]
            dims = effective_leanvec_dims(leanvec_dims, dim)
            if train_queries is None:
                return _call(
                    "svs_storage_create_leanvec", dims, primary, secondary
                )
            train_vectors = _as_float32(train_vectors)
            train_queries = _as_float32(train_queries)
            training_data = _call(
                "svs_leanvec_training_data_build",
                self.handle,
                dims,
                train_vectors.shape[0],
                train_vectors.ctypes.data,
                train_queries.shape[0],
                train_queries.ctypes.data,
            )
            try:
                return _call(
                    "svs_storage_create_leanvec_trained",
                    training_data,
                    primary,
                    secondary,
                )
            finally:
                lib().svs_leanvec_training_data_free(training_data)
        raise ValueError(f"Unknown {svs_type=}")

    def set_threadpool(self, num_threads: int, threadpool: str) -> None:
        """Set the thread pool of indices created by this builder.

        threadpool is "native" or "omp". A native pool with one thread
        is replaced by running every call on its caller's thread,
        because a native pool serializes concurrent calls unless the
        library wraps it for SVS_SYNC_KIND_FINE_GRAIN.
        """
        if threadpool == "omp":
            kind = _ThreadpoolKind.OMP
        elif threadpool != "native":
            raise ValueError(f"Unknown {threadpool=}")
        elif num_threads == 1:
            kind = _ThreadpoolKind.SINGLE_THREAD
        else:
            kind = _ThreadpoolKind.NATIVE
        _call(
            "svs_index_builder_set_threadpool", self.handle, kind, num_threads
        )

    def free(self) -> None:
        while self._handles:
            function, handle = self._handles.pop()
            getattr(lib(), function)(handle)


def _dynamic_params(sync_kind: SyncKind) -> _DynamicIndexParams:
    return _DynamicIndexParams(
        version=_HEADER_VERSION,
        struct_size=ctypes.sizeof(_DynamicIndexParams),
        blocksize_bytes=0,
        blocksize_elements=0,
        sync_kind=sync_kind,
    )


class DynamicIndex:
    """Dynamic Vamana index of the SVS C API."""

    def __init__(self, builder: _Builder, handle: int, dim: int) -> None:
        self._builder = builder
        self.handle = handle
        self.dim = dim

    @classmethod
    def build(
        cls,
        vectors: npt.ArrayLike,
        ids: npt.ArrayLike | None = None,
        *,
        svs_type: str = "float32",
        distance: str = "mip",
        graph_max_degree: int = 64,
        window_size: int = 200,
        alpha: float | None = None,
        num_threads: int = 1,
        threadpool: str = "native",
        leanvec_dims: int = -4,
        train_vectors: npt.NDArray | None = None,
        train_queries: npt.NDArray | None = None,
        sync_kind: SyncKind = SyncKind.FINE_GRAIN,
    ) -> "DynamicIndex":
        """Build a dynamic index from vectors.

        If train_queries is given, the LeanVec matrices are trained
        out-of-distribution on train_vectors (default: vectors) and
        train_queries. Otherwise, LeanVec uses PCA of vectors.
        """
        vectors = _as_float32(vectors)
        dim = vectors.shape[1]
        builder = _Builder(
            svs_type=svs_type,
            distance=distance,
            dim=dim,
            graph_max_degree=graph_max_degree,
            window_size=window_size,
            alpha=alpha,
            num_threads=num_threads,
            threadpool=threadpool,
            leanvec_dims=leanvec_dims,
            train_vectors=vectors if train_vectors is None else train_vectors,
            train_queries=train_queries,
        )
        try:
            ids_array = None if ids is None else _as_ids(ids)
            params = _dynamic_params(sync_kind)
            handle = _call(
                "svs_index_build_dynamic_ex",
                builder.handle,
                vectors.ctypes.data,
                None if ids_array is None else ids_array.ctypes.data,
                vectors.shape[0],
                ctypes.byref(params),
            )
        except BaseException:
            builder.free()
            raise
        return cls(builder, handle, dim)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        dim: int,
        svs_type: str = "float32",
        distance: str = "mip",
        num_threads: int = 1,
        threadpool: str = "native",
        leanvec_dims: int = -4,
        sync_kind: SyncKind = SyncKind.FINE_GRAIN,
    ) -> "DynamicIndex":
        """Load a dynamic index saved with save()."""
        builder = _Builder(
            svs_type=svs_type,
            distance=distance,
            dim=dim,
            graph_max_degree=64,
            window_size=200,
            alpha=None,
            num_threads=num_threads,
            threadpool=threadpool,
            leanvec_dims=leanvec_dims,
        )
        try:
            params = _dynamic_params(sync_kind)
            handle = _call(
                "svs_index_load_dynamic_ex",
                builder.handle,
                str(directory).encode(),
                ctypes.byref(params),
            )
        except BaseException:
            builder.free()
            raise
        return cls(builder, handle, dim)

    def convert(
        self,
        *,
        svs_type: str,
        distance: str = "mip",
        graph_max_degree: int = 64,
        window_size: int = 200,
        alpha: float | None = None,
        num_threads: int = 1,
        threadpool: str = "native",
        leanvec_dims: int = -4,
        train_vectors: npt.NDArray | None = None,
        train_queries: npt.NDArray | None = None,
        sync_kind: SyncKind = SyncKind.FINE_GRAIN,
    ) -> "DynamicIndex":
        """Convert to a new dynamic index with another storage type.

        If train_queries is given, the LeanVec matrices are trained
        out-of-distribution on train_vectors and train_queries.
        Otherwise, LeanVec uses PCA of the data of this index.
        """
        if (train_vectors is None) != (train_queries is None):
            raise ValueError("Pass both or none of train_vectors/queries")
        builder = _Builder(
            svs_type=svs_type,
            distance=distance,
            dim=self.dim,
            graph_max_degree=graph_max_degree,
            window_size=window_size,
            alpha=alpha,
            num_threads=num_threads,
            threadpool=threadpool,
            leanvec_dims=leanvec_dims,
            train_vectors=train_vectors,
            train_queries=train_queries,
        )
        try:
            params = _dynamic_params(sync_kind)
            handle = _call(
                "svs_index_convert_dynamic_ex",
                builder.handle,
                self.handle,
                ctypes.byref(params),
            )
        except BaseException:
            builder.free()
            raise
        return type(self)(builder, handle, self.dim)

    def __enter__(self) -> "DynamicIndex":
        return self

    def __exit__(self, *_) -> None:
        self.free()

    def free(self) -> None:
        if self.handle is not None:
            lib().svs_index_free(self.handle)
            self.handle = None
            self._builder.free()

    @property
    def size(self) -> int:
        size = ctypes.c_size_t()
        _call("svs_index_get_size", self.handle, ctypes.byref(size))
        return size.value

    @property
    def num_threads(self) -> int:
        num_threads = ctypes.c_size_t()
        _call(
            "svs_index_get_num_threads", self.handle, ctypes.byref(num_threads)
        )
        return num_threads.value

    @num_threads.setter
    def num_threads(self, num_threads: int) -> None:
        _call("svs_index_set_num_threads", self.handle, num_threads)

    @property
    def sync_kind(self) -> SyncKind:
        sync_kind = ctypes.c_uint32()
        _call(
            "svs_index_dynamic_get_sync_kind",
            self.handle,
            ctypes.byref(sync_kind),
        )
        return SyncKind(sync_kind.value)

    def save(self, directory: Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        _call("svs_index_save", self.handle, str(directory).encode())

    def add(self, vectors: npt.NDArray, ids: npt.NDArray) -> None:
        """Add vectors in a single call.

        Use add_address() in hot loops to avoid conversions.
        """
        vectors = _as_float32(vectors)
        ids = _as_ids(ids)
        self.add_address(vectors.ctypes.data, ids.ctypes.data, len(ids))

    def add_address(self, vectors: int, ids: int, num_vectors: int) -> None:
        """Add vectors given addresses of float32 and size_t arrays."""
        err = _error()
        if not lib().svs_index_dynamic_add_points(
            self.handle, vectors, ids, num_vectors, None, err
        ):
            _raise("svs_index_dynamic_add_points", err)

    def delete(self, ids: npt.NDArray) -> None:
        ids = _as_ids(ids)
        self.delete_address(ids.ctypes.data, len(ids))

    def delete_address(self, ids: int, num_ids: int) -> None:
        """Delete vectors given the address of a size_t array."""
        err = _error()
        if not lib().svs_index_dynamic_delete_points(
            self.handle, ids, num_ids, None, err
        ):
            _raise("svs_index_dynamic_delete_points", err)


class SearchParams:
    """Vamana search parameters."""

    def __init__(self, search_window_size: int) -> None:
        self.search_window_size = search_window_size
        self.handle = _call(
            "svs_search_params_create_vamana", search_window_size
        )

    def __enter__(self) -> "SearchParams":
        return self

    def __exit__(self, *_) -> None:
        self.free()

    def free(self) -> None:
        if self.handle is not None:
            lib().svs_search_params_free(self.handle)
            self.handle = None


class Searcher:
    """Searches with caller-owned result buffers.

    A Searcher must not be used by multiple threads at the same time.
    Results of a call to
    search_address() are written to rows [first, first + num_queries)
    of the indices and distances arrays given at construction.
    """

    def __init__(
        self,
        index: DynamicIndex,
        k: int,
        max_queries_per_call: int,
        indices: npt.NDArray[np.uintp],
        distances: npt.NDArray[np.float32],
        search_params: SearchParams | None = None,
    ) -> None:
        if indices.dtype != np.uintp or distances.dtype != np.float32:
            raise ValueError("indices must be uintp and distances float32")
        self._index = index.handle
        self._k = k
        self._dim = index.dim
        self._offsets = np.zeros(max_queries_per_call + 1, dtype=np.uintp)
        self._indices = indices
        self._distances = distances
        self._results = _SearchResults(
            version=_HEADER_VERSION,
            struct_size=ctypes.sizeof(_SearchResults),
            offsets=self._offsets.ctypes.data,
            offsets_capacity=len(self._offsets),
            results_capacity=max_queries_per_call * k,
            owns_buffers=False,
        )
        self._results_ref = ctypes.byref(self._results)
        self._indices_address = indices.ctypes.data
        self._distances_address = distances.ctypes.data
        self._row_bytes_indices = indices.strides[0]
        self._row_bytes_distances = distances.strides[0]
        self._params = None if search_params is None else search_params.handle
        self._function = lib().svs_index_search_topk

    def search_address(self, queries: int, first: int, num_queries: int):
        """Search queries given the address of a float32 array."""
        err = _error()
        results = self._results
        results.indices = (
            self._indices_address + first * self._row_bytes_indices
        )
        results.distances = (
            self._distances_address + first * self._row_bytes_distances
        )
        if not self._function(
            self._index,
            queries,
            num_queries,
            self._k,
            self._results_ref,
            self._params,
            None,
            err,
        ):
            _raise("svs_index_search_topk", err)
        if results.total_results != num_queries * self._k:
            raise CAPIError(
                "svs_index_search_topk",
                0,
                f"expected {num_queries * self._k} results"
                f" but got {results.total_results}",
            )


def runtime_info() -> dict[str, dict[str, object]]:
    """Return C API and Python runtime information for logging."""
    import sys

    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "capi_runtime": {
            "library_path": library_path(),
            "library_version": version(),
            "python_version": sys.version,
            "gil_enabled": gil_enabled,
        }
    }
