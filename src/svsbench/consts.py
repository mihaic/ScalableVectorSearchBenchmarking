# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Constants."""

from typing import Final

import numpy as np
import svs

DISTANCE_TO_ALPHA: Final[dict[svs.DistanceType, float]] = {
    svs.DistanceType.Cosine: 0.95,
    svs.DistanceType.L2: 1.2,
    svs.DistanceType.MIP: 0.95,
}

STR_TO_DISTANCE: Final[dict[str, svs.DistanceType]] = {
    "cosine": svs.DistanceType.Cosine,
    "l2": svs.DistanceType.L2,
    "mip": svs.DistanceType.MIP,
}

DISTANCE_TO_STR: Final[dict[svs.DistanceType, str]] = {
    v: k for k, v in STR_TO_DISTANCE.items()
}

SUFFIX_TO_DATA_TYPE: Final = {
    ".fvecs": svs.DataType.float32,
    ".hvecs": svs.DataType.float16,
    ".ivecs": svs.DataType.uint32,
    ".int8vecs": svs.DataType.int8,
}

SUFFIX_TO_SVS_TYPE: Final = {
    ".hvecs": "float16",
    ".fvecs": "float32",
    ".int8vecs": "int8",
}

SVS_TYPES: Final = (
    "float16",
    "float32",
    "int8",
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
    "int8": np.int8,
}

STR_TO_DATA_TYPE: Final = {
    "float16": svs.DataType.float16,
    "float32": svs.DataType.float32,
    "int8": svs.DataType.int8,
}

STR_TO_LVQ_STRATEGY: Final[dict[str, svs.LVQStrategy]] = {
    "auto": svs.LVQStrategy.Auto,
    "sequential": svs.LVQStrategy.Sequential,
    "turbo": svs.LVQStrategy.Turbo,
}
