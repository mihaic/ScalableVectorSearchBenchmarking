// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! SVS C API benchmarks in Rust, mirroring `svsbench.build_capi` and
//! `svsbench.search_capi`.

pub mod capi;
pub mod utils;

pub const SVS_TYPES: [&str; 8] = [
    "float32", "float16", "lvq8", "lvq4x4", "lvq4x8", "leanvec4x4", "leanvec4x8", "leanvec8x8",
];
pub const DISTANCES: [&str; 3] = ["cosine", "l2", "mip"];
pub const DEFAULT_LEANVEC_DIMS: i64 = -4;
pub const DEFAULT_LEANVEC_TRAIN_MAX_VECTORS: usize = 100_000;

pub fn default_alpha(distance: &str) -> f32 {
    if distance == "l2" { 1.2 } else { 0.95 }
}
