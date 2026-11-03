// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! Search benchmark using the SVS C API.
//!
//! Rust port of `svsbench.search_capi`, with the same options except
//! --shuffle. The index is loaded with SVS_SYNC_KIND_FINE_GRAIN. Queries
//! are searched by max_threads threads concurrently, call_size queries
//! per C API call. The threads persist across repetitions, like the
//! Python ThreadPoolExecutor.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Parser;
use clap::builder::PossibleValuesParser;
use serde_json::json;

use svsbench::capi::{self, BuilderConfig, DynamicIndex, SearchParams, Searcher, SyncKind};
use svsbench::utils::{self, Logger};
use svsbench::{DEFAULT_LEANVEC_DIMS, DISTANCES, SVS_TYPES};

#[derive(Parser, Debug)]
#[command(about = "Search benchmark using the SVS C API")]
struct Args {
    /// Index dir
    #[arg(long = "idx_dir")]
    idx_dir: PathBuf,
    /// Query fvecs file
    #[arg(long = "query_file")]
    query_file: PathBuf,
    /// Ground truth ivecs file
    #[arg(long = "ground_truth_file")]
    ground_truth_file: PathBuf,
    /// Search window size (repeat to search with several)
    #[arg(long = "search_window_size", required = true)]
    search_window_size: Vec<usize>,
    /// Output dir
    #[arg(long = "out_dir", default_value = "out")]
    out_dir: PathBuf,
    /// Log dir [default: out_dir]
    #[arg(long = "log_dir")]
    log_dir: Option<PathBuf>,
    #[arg(long = "svs_type", default_value = "float32", value_parser = PossibleValuesParser::new(SVS_TYPES))]
    svs_type: String,
    #[arg(long = "distance", default_value = "mip", value_parser = PossibleValuesParser::new(DISTANCES))]
    distance: String,
    /// Number of neighbors to return
    #[arg(short = 'k', default_value_t = 10)]
    k: usize,
    /// Number of threads calling the C API [default: CPUs - 1]
    #[arg(long = "max_threads")]
    max_threads: Option<usize>,
    /// Number of index threads. With 1, each search runs on its calling thread
    #[arg(long = "pool_threads", default_value_t = 1)]
    pool_threads: usize,
    /// Index thread pool kind (omp requires a library built with OpenMP)
    #[arg(long = "threadpool", default_value = "native", value_parser = PossibleValuesParser::new(capi::THREADPOOLS))]
    threadpool: String,
    /// Number of queries per C API call
    #[arg(long = "call_size", default_value_t = 1)]
    call_size: usize,
    /// Number of search repetitions
    #[arg(long = "num_rep", default_value_t = 5)]
    num_rep: usize,
    /// LeanVec dimensionality (negative: divisor of dimensionality)
    #[arg(long = "leanvec_dims", default_value_t = DEFAULT_LEANVEC_DIMS, allow_negative_numbers = true)]
    leanvec_dims: i64,
    /// Run even if there are changes in SVSBench not committed to git
    #[arg(long = "uncommitted")]
    uncommitted: bool,
}

/// Pointer to a buffer whose disjoint parts are written by several threads.
#[derive(Clone, Copy)]
struct SharedMut<T>(*mut T);
unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (logger, log_file) = Logger::new("search_capi", args.log_dir.as_ref().unwrap_or(&args.out_dir))?;
    println!("Logging to\n{}", log_file.display());
    logger.info(json!({"argv": std::env::args().collect::<Vec<_>>()}));
    utils::check_uncommitted_and_log_version(&logger, args.uncommitted)?;
    logger.info(json!({"search_args": format!("{args:?}")}));
    logger.info(utils::read_system_config());
    logger.info(capi::runtime_info());

    let (query, dim) = utils::read_vectors(&args.query_file, None)?;
    let (ground_truth, gt_dim) = utils::read_vecs::<u32>(&args.ground_truth_file, None)?;
    let k = args.k;
    if gt_dim < k {
        bail!("Ground truth has {gt_dim} neighbors, need {k}");
    }
    let num_queries = query.len() / dim;
    let call_size = args.call_size;
    let num_calls = num_queries.div_ceil(call_size);
    let max_threads = args.max_threads.unwrap_or_else(utils::default_max_threads).min(num_calls);
    let num_rep = args.num_rep;

    let config = BuilderConfig {
        svs_type: args.svs_type.clone(),
        distance: args.distance.clone(),
        graph_max_degree: 64,
        window_size: 200,
        alpha: None,
        num_threads: args.pool_threads,
        threadpool: args.threadpool.clone(),
        leanvec_dims: args.leanvec_dims,
    };
    let index = DynamicIndex::load(&args.idx_dir, dim, &config, SyncKind::FineGrain)?;
    logger.info(json!({"sync_kind": index.sync_kind()?.name()}));

    let mut results = vec![0usize; num_queries * k];
    let mut distances = vec![0f32; num_queries * k];
    let mut latencies = vec![0f64; num_calls];
    let results_ptr = SharedMut(results.as_mut_ptr());
    let distances_ptr = SharedMut(distances.as_mut_ptr());
    let latencies_ptr = SharedMut(latencies.as_mut_ptr());

    // Workers wait at `start`, search all queries with the parameters in
    // `params`, and meet the main thread at `done`.
    let start = Barrier::new(max_threads + 1);
    let done = Barrier::new(max_threads + 1);
    let params = AtomicPtr::<SearchParams>::new(std::ptr::null_mut());
    let stop = AtomicBool::new(false);
    let error = Mutex::new(None);

    std::thread::scope(|scope| -> Result<()> {
        for thread_idx in 0..max_threads {
            let (index, query, start, done, params, stop, error) =
                (&index, &query, &start, &done, &params, &stop, &error);
            scope.spawn(move || {
                let mut searcher = Searcher::new(index, k, call_size);
                loop {
                    start.wait();
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    // SAFETY: the main thread keeps the parameters alive
                    // until all threads reach `done`.
                    let params = unsafe { &*params.load(Ordering::Acquire) };
                    for call_idx in (thread_idx..num_calls).step_by(max_threads) {
                        let first = call_idx * call_size;
                        let num = call_size.min(num_queries - first);
                        // SAFETY: each call writes rows [first, first + num)
                        // and latency call_idx, which no other call writes,
                        // and the main thread reads them only after `done`.
                        let (indices, distances) = unsafe {
                            (
                                std::slice::from_raw_parts_mut(results_ptr.get().add(first * k), num * k),
                                std::slice::from_raw_parts_mut(distances_ptr.get().add(first * k), num * k),
                            )
                        };
                        let queries = &query[first * dim..(first + num) * dim];
                        let call_start = Instant::now();
                        let result = searcher.search(queries, indices, distances, params);
                        unsafe { *latencies_ptr.get().add(call_idx) = call_start.elapsed().as_secs_f64() };
                        if let Err(e) = result {
                            error.lock().unwrap().get_or_insert(e);
                            break;
                        }
                    }
                    done.wait();
                }
            });
        }

        let run = || -> Result<f64> {
            let run_start = Instant::now();
            start.wait();
            done.wait();
            let total_time = run_start.elapsed().as_secs_f64();
            match error.lock().unwrap().take() {
                Some(e) => Err(e),
                None => Ok(total_time),
            }
        };
        let outcome = (|| -> Result<()> {
            for &search_window_size in &args.search_window_size {
                let mut search_params = SearchParams::new(search_window_size)?;
                params.store(&mut search_params, Ordering::Release);
                // Warm-up search
                run()?;
                let (mut qps, mut p95s, mut p99s, mut mean_latencies) = (vec![], vec![], vec![], vec![]);
                for _ in 0..num_rep {
                    let total_time = run()?;
                    // SAFETY: the workers are waiting at `start`.
                    let latencies = unsafe { std::slice::from_raw_parts(latencies_ptr.get(), num_calls) };
                    qps.push(num_queries as f64 / total_time);
                    p95s.push(utils::percentile(latencies, 95.0));
                    p99s.push(utils::percentile(latencies, 99.0));
                    mean_latencies.push(utils::mean(latencies));
                }
                params.store(std::ptr::null_mut(), Ordering::Release);
                drop(search_params);
                let results = unsafe { std::slice::from_raw_parts(results_ptr.get(), num_queries * k) };
                let recall = utils::k_recall_at(&ground_truth, gt_dim, results, k, k, k);
                logger.info(json!({"search_results": {
                    "qps": qps,
                    "qps_mean": utils::mean(&qps),
                    "qps_rsd": utils::std_ddof(&qps) / utils::mean(&qps),
                    "p95": p95s,
                    "p95_mean": utils::mean(&p95s),
                    "p95_rsd": utils::std_ddof(&p95s) / utils::mean(&p95s),
                    "p99_mean": utils::mean(&p99s),
                    "latency_mean": utils::mean(&mean_latencies),
                    "search_parameters": {"search_window_size": search_window_size},
                    "batch_size": call_size,
                    "max_threads": max_threads,
                    "recall": recall,
                }}));
            }
            Ok(())
        })();
        stop.store(true, Ordering::Release);
        start.wait();
        outcome
    })?;
    Ok(())
}
