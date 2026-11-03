# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Constants.

Constants that depend on the SVS Python bindings are created lazily on
first access so that modules using only the C API can be imported
without the bindings, e.g., on free-threaded Python.
"""

from typing import Any, Final

import numpy as np

DEFAULT_LEANVEC_DIMS: Final[int] = -4
DEFAULT_LEANVEC_TRAIN_MAX_VECTORS: Final[int] = 100_000

DISTANCES: Final = ("cosine", "l2", "mip")

STR_TO_ALPHA: Final[dict[str, float]] = {
    "cosine": 0.95,
    "l2": 1.2,
    "mip": 0.95,
}

SUFFIX_TO_SVS_TYPE: Final = {
    ".hvecs": "float16",
    ".fvecs": "float32",
}

SVS_TYPES: Final = (
    "float16",
    "float32",
    "leanvec4x4",
    "leanvec4x8",
    "leanvec8x8",
    "lvq8",
    "lvq4x4",
    "lvq4x8",
)

SVS_TYPE_TO_DTYPE: Final = {
    "float16": np.float16,
    "float32": np.float32,
}


def _svs_constants() -> dict[str, Any]:
    import svs

    str_to_distance = {
        "cosine": svs.DistanceType.Cosine,
        "l2": svs.DistanceType.L2,
        "mip": svs.DistanceType.MIP,
    }
    return {
        "DISTANCE_TO_ALPHA": {
            str_to_distance[k]: v for k, v in STR_TO_ALPHA.items()
        },
        "STR_TO_DISTANCE": str_to_distance,
        "DISTANCE_TO_STR": {v: k for k, v in str_to_distance.items()},
        "SUFFIX_TO_DATA_TYPE": {
            ".fvecs": svs.DataType.float32,
            ".hvecs": svs.DataType.float16,
            ".ivecs": svs.DataType.uint32,
        },
        "STR_TO_DATA_TYPE": {
            "float16": svs.DataType.float16,
            "float32": svs.DataType.float32,
        },
        "STR_TO_LVQ_STRATEGY": {
            "auto": svs.LVQStrategy.Auto,
            "sequential": svs.LVQStrategy.Sequential,
            "turbo": svs.LVQStrategy.Turbo,
        },
    }


_SVS_CONSTANT_NAMES: Final = frozenset(
    (
        "DISTANCE_TO_ALPHA",
        "STR_TO_DISTANCE",
        "DISTANCE_TO_STR",
        "SUFFIX_TO_DATA_TYPE",
        "STR_TO_DATA_TYPE",
        "STR_TO_LVQ_STRATEGY",
    )
)


def __getattr__(name: str) -> Any:
    if name in _SVS_CONSTANT_NAMES:
        constants = _svs_constants()
        globals().update(constants)
        return constants[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
