// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! Logging, I/O, threading and statistics helpers mirroring
//! `svsbench/utils.py` and `svsbench/merge.py`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use chrono::{Local, SecondsFormat};
use serde_json::{Value, json};

pub const VERSION: &str = env!("SVSBENCH_GIT_VERSION");

/// Logger writing JSON lines `[msg, timestamp]` like the Python JSONFormatter.
pub struct Logger {
    file: Mutex<BufWriter<File>>,
}

fn timestamp() -> String {
    Local::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

impl Logger {
    /// Create log_dir/<name>-<timestamp>.log.
    pub fn new(name: &str, log_dir: &Path) -> Result<(Self, PathBuf)> {
        std::fs::create_dir_all(log_dir)?;
        let path = log_dir.join(format!("{name}-{}.log", timestamp()));
        let file = File::create(&path).with_context(|| path.display().to_string())?;
        Ok((
            Self {
                file: Mutex::new(BufWriter::new(file)),
            },
            path,
        ))
    }

    pub fn info(&self, msg: Value) {
        let line = json!([msg, timestamp()]).to_string();
        eprintln!("{line}");
        let mut file = self.file.lock().unwrap();
        writeln!(file, "{line}").and_then(|()| file.flush()).expect("log write");
    }
}

/// Fail on uncommitted changes unless allowed and log the version.
pub fn check_uncommitted_and_log_version(logger: &Logger, uncommitted: bool) -> Result<()> {
    if VERSION.ends_with("-dirty") {
        if !uncommitted {
            bail!(
                "Uncommitted changes detected. To ignore this error, \
                 add --uncommitted to the SVSBench call."
            );
        }
        logger.info(json!("uncommitted_code_changes"));
    }
    logger.info(json!({"version": VERSION}));
    Ok(())
}

pub fn read_system_config() -> Value {
    let read = |path: &str| std::fs::read_to_string(path).ok().map(|s| s.trim().to_owned());
    let mut free_huge_pages = Vec::new();
    if let Ok(nodes) = std::fs::read_dir("/sys/devices/system/node/") {
        let mut nodes: Vec<_> = nodes
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("node"))
            .map(|entry| entry.path())
            .collect();
        nodes.sort();
        for node in nodes {
            let path = node.join("hugepages/hugepages-1048576kB/free_hugepages");
            if let Some(n) = read(&path.to_string_lossy()).and_then(|s| s.parse::<u64>().ok()) {
                free_huge_pages.push(n);
            }
        }
    }
    json!({
        "system_config": {
            "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo")
                .and_then(|s| s.parse::<i64>().ok()),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "free_huge_pages": free_huge_pages,
        }
    })
}

pub fn default_max_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Element of a *vecs file: 4 bytes, any bit pattern valid.
pub trait VecsElement: Copy + Default {}
impl VecsElement for f32 {}
impl VecsElement for u32 {}

/// Read up to max_vectors (all if None) rows of an fvecs or ivecs file.
///
/// Returns the row-major data and the dimensionality.
pub fn read_vecs<T: VecsElement>(path: &Path, max_vectors: Option<usize>) -> Result<(Vec<T>, usize)> {
    let file = File::open(path).with_context(|| path.display().to_string())?;
    let len = file.metadata()?.len() as usize;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let dim = u32::from_le_bytes(header) as usize;
    let row_bytes = 4 * (dim + 1);
    if dim == 0 || len % row_bytes != 0 {
        bail!("{}: size {len} is not a multiple of rows of dim {dim}", path.display());
    }
    let rows = (len / row_bytes).min(max_vectors.unwrap_or(usize::MAX));
    let mut data = vec![T::default(); rows * dim];
    for (i, row) in data.chunks_exact_mut(dim).enumerate() {
        if i > 0 {
            reader.read_exact(&mut header)?;
        }
        // SAFETY: T is 4 bytes with any bit pattern valid.
        let bytes = unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr().cast::<u8>(), dim * 4) };
        reader.read_exact(bytes)?;
    }
    Ok((data, dim))
}

/// Read float32 vectors from an fvecs file.
pub fn read_vectors(path: &Path, max_vectors: Option<usize>) -> Result<(Vec<f32>, usize)> {
    if path.extension().is_none_or(|ext| ext != "fvecs") {
        bail!("Only .fvecs vectors are supported: {}", path.display());
    }
    read_vecs(path, max_vectors)
}

/// Write a 1-D float64 array in NumPy .npy format.
pub fn save_npy(path: &Path, values: &[f64]) -> Result<()> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    // Magic (6) + version (2) + header length (2) + header must be a
    // multiple of 64, with the header ending in a newline.
    let padding = 63 - (10 + header.len()) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&(header.len() as u16).to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

/// Call function(first, count) on num_threads scoped threads.
///
/// The items [0, num_items) are split into calls of at most
/// items_per_call items, assigned round-robin to the threads.
pub fn run_in_threads<F>(num_threads: usize, num_items: usize, items_per_call: usize, function: F) -> Result<()>
where
    F: Fn(usize, usize) -> Result<()> + Sync,
{
    let num_calls = num_items.div_ceil(items_per_call);
    let num_threads = num_threads.min(num_calls);
    let function = &function;
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..num_threads)
            .map(|thread_idx| {
                scope.spawn(move || {
                    for call_idx in (thread_idx..num_calls).step_by(num_threads) {
                        let first = call_idx * items_per_call;
                        function(first, items_per_call.min(num_items - first))?;
                    }
                    Ok(())
                })
            })
            .collect();
        workers.into_iter().try_for_each(|worker| worker.join().unwrap())
    })
}

/// Compute k-recall@at like svs.k_recall_at for rows without duplicates.
pub fn k_recall_at(ground_truth: &[u32], gt_dim: usize, results: &[usize], results_dim: usize, k: usize, at: usize) -> f64 {
    assert!(k <= at && k <= gt_dim && at <= results_dim);
    let num_queries = results.len() / results_dim;
    let matches: usize = (0..num_queries)
        .map(|q| {
            let result = &results[q * results_dim..q * results_dim + at];
            ground_truth[q * gt_dim..q * gt_dim + k]
                .iter()
                .filter(|&&id| result.contains(&(id as usize)))
                .count()
        })
        .sum();
    matches as f64 / (num_queries * k) as f64
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Standard deviation with ddof=min(1, n - 1), like the Python code.
pub fn std_ddof(values: &[f64]) -> f64 {
    let ddof = 1.min(values.len() - 1);
    let mean = mean(values);
    let ss: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    (ss / (values.len() - ddof) as f64).sqrt()
}

/// Percentile with linear interpolation, like numpy.percentile.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = q / 100.0 * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_match_numpy() {
        let values = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert!((percentile(&values, 95.0) - 7.4).abs() < 1e-12);
        assert!((std_ddof(&values) - 3.095695936834452).abs() < 1e-12);
        assert_eq!(std_ddof(&[5.0]), 0.0);
    }

    #[test]
    fn recall() {
        let ground_truth = [0, 1, 9, 2, 3, 9];
        let results = [1, 5, 2, 7];
        assert_eq!(k_recall_at(&ground_truth, 3, &results, 2, 2, 2), 0.5);
    }
}
