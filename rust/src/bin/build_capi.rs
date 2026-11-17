// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//! Benchmark SVS dynamic index addition using the SVS C API.
//!
//! Rust port of `svsbench.build_capi`, with the same options except
//! --shuffle and --num_vectors_delete, and with --sync_kind and
//! --convert_after_add. Vectors are added by max_threads threads
//! concurrently, call_size vectors per C API call.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use clap::builder::PossibleValuesParser;
use serde_json::json;

use svsbench::capi::{self, BuilderConfig, DynamicIndex, Matrix, SyncKind};
use svsbench::utils::{self, Logger};
use svsbench::{DEFAULT_LEANVEC_DIMS, DEFAULT_LEANVEC_TRAIN_MAX_VECTORS, DISTANCES, SVS_TYPES};

#[derive(Parser, Debug)]
#[command(about = "Benchmark SVS dynamic index addition using the SVS C API")]
struct Args {
    /// Vectors fvecs file
    #[arg(long = "vecs_file")]
    vecs_file: PathBuf,
    /// Output dir
    #[arg(long = "out_dir", default_value = "out")]
    out_dir: PathBuf,
    /// Log dir [default: out_dir]
    #[arg(long = "log_dir")]
    log_dir: Option<PathBuf>,
    /// Index dir to load instead of building
    #[arg(long = "idx_dir")]
    idx_dir: Option<PathBuf>,
    #[arg(long = "svs_type", default_value = "float32", value_parser = PossibleValuesParser::new(SVS_TYPES))]
    svs_type: String,
    #[arg(long = "distance", default_value = "mip", value_parser = PossibleValuesParser::new(DISTANCES))]
    distance: String,
    /// Number of vectors
    #[arg(long = "num_vectors")]
    num_vectors: Option<usize>,
    /// Number of vectors added per timed batch
    #[arg(long = "batch_size", default_value_t = 10000)]
    batch_size: usize,
    /// Number of vectors to add initially [default: batch_size]
    #[arg(long = "num_vectors_init")]
    num_vectors_init: Option<usize>,
    /// Proportion of vectors to add initially
    #[arg(long = "proportion_vectors_init")]
    proportion_vectors_init: Option<f64>,
    #[arg(long = "graph_max_degree", default_value_t = 64)]
    graph_max_degree: usize,
    #[arg(long = "window_size", default_value_t = 200)]
    window_size: usize,
    #[arg(long = "alpha")]
    alpha: Option<f32>,
    /// Number of threads calling the C API [default: CPUs - 1]
    #[arg(long = "max_threads")]
    max_threads: Option<usize>,
    /// Number of index threads for building the initial graph [default: max_threads]
    #[arg(long = "max_threads_init")]
    max_threads_init: Option<usize>,
    /// Number of index threads after the initial build [default: max_threads]
    #[arg(long = "pool_threads")]
    pool_threads: Option<usize>,
    /// Index thread pool kind (omp requires a library built with OpenMP)
    #[arg(long = "threadpool", default_value = "native", value_parser = PossibleValuesParser::new(capi::THREADPOOLS))]
    threadpool: String,
    /// Number of vectors per C API call
    #[arg(long = "call_size", default_value_t = 1)]
    call_size: usize,
    /// LeanVec dimensionality (negative: divisor of dimensionality)
    #[arg(long = "leanvec_dims", default_value_t = DEFAULT_LEANVEC_DIMS, allow_negative_numbers = true)]
    leanvec_dims: i64,
    /// Query fvecs file for LeanVec out-of-distribution training
    #[arg(long = "train_query_file")]
    train_query_file: Option<PathBuf>,
    /// Maximum number of base vectors from vecs_file to use for LeanVec
    /// out-of-distribution training (0 for all)
    #[arg(long = "train_max_vectors", default_value_t = DEFAULT_LEANVEC_TRAIN_MAX_VECTORS)]
    train_max_vectors: usize,
    /// Build the initial index with float32 and convert it to svs_type
    /// with svs_index_convert_dynamic_ex
    #[arg(long = "convert")]
    convert: bool,
    /// With --convert, convert after adding all vectors instead of after
    /// building the initial index
    #[arg(long = "convert_after_add", requires = "convert")]
    convert_after_add: bool,
    /// Index synchronization kind (NONE requires max_threads 1)
    #[arg(long = "sync_kind", default_value = "FINE_GRAIN", value_parser = PossibleValuesParser::new(["NONE", "GLOBAL", "FINE_GRAIN"]))]
    sync_kind: String,
    /// Do not save built index
    #[arg(long = "no_save")]
    no_save: bool,
    /// Run even if there are changes in SVSBench not committed to git
    #[arg(long = "uncommitted")]
    uncommitted: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (logger, log_file) = Logger::new("build_capi", args.log_dir.as_ref().unwrap_or(&args.out_dir))?;
    println!("Logging to\n{}", log_file.display());
    logger.info(json!({"argv": std::env::args().collect::<Vec<_>>()}));
    utils::check_uncommitted_and_log_version(&logger, args.uncommitted)?;
    std::fs::create_dir_all(&args.out_dir)?;
    logger.info(json!({"build_args": format!("{args:?}")}));
    logger.info(utils::read_system_config());
    logger.info(capi::runtime_info());

    let (vectors, dim) = utils::read_vectors(&args.vecs_file, args.num_vectors)?;
    let num_vectors = vectors.len() / dim;
    if args.num_vectors.is_some_and(|n| n > num_vectors) {
        anyhow::bail!("num_vectors is greater than the number of vectors, {num_vectors}");
    }
    let num_vectors_init = args.num_vectors_init.unwrap_or_else(|| match args.proportion_vectors_init {
        Some(proportion) => (num_vectors as f64 * proportion) as usize,
        None => args.batch_size,
    });
    let max_threads = args.max_threads.unwrap_or_else(utils::default_max_threads);
    let sync_kind = match args.sync_kind.as_str() {
        "NONE" => SyncKind::None,
        "GLOBAL" => SyncKind::Global,
        _ => SyncKind::FineGrain,
    };
    if sync_kind == SyncKind::None && max_threads != 1 {
        anyhow::bail!("sync_kind NONE requires max_threads 1");
    }
    let convert_init = args.convert && !args.convert_after_add;
    let max_threads_init = args.max_threads_init.unwrap_or(max_threads);
    let pool_threads = args.pool_threads.unwrap_or(max_threads);
    let train_queries = match &args.train_query_file {
        Some(path) => Some(utils::read_vectors(path, None)?),
        None => None,
    };
    let training = train_queries.as_ref().map(|(queries, query_dim)| {
        let train_max = match args.train_max_vectors {
            0 => num_vectors,
            n => n.min(num_vectors),
        };
        (
            Matrix { data: &vectors[..train_max * dim], dim },
            Matrix { data: queries, dim: *query_dim },
        )
    });
    let config = BuilderConfig {
        svs_type: args.svs_type.clone(),
        distance: args.distance.clone(),
        graph_max_degree: args.graph_max_degree,
        window_size: args.window_size,
        alpha: Some(args.alpha.unwrap_or_else(|| svsbench::default_alpha(&args.distance))),
        num_threads: max_threads_init,
        threadpool: args.threadpool.clone(),
        leanvec_dims: args.leanvec_dims,
    };

    let num_batches = (num_vectors - num_vectors_init).div_ceil(args.batch_size);
    let mut ingest_time = vec![0.0; num_batches + 1];
    let vector_ids: Vec<usize> = (0..num_vectors).collect();

    let mut index = match &args.idx_dir {
        None => {
            let init = Matrix { data: &vectors[..num_vectors_init * dim], dim };
            let start = Instant::now();
            let mut index = if args.convert {
                let float32 = BuilderConfig { svs_type: "float32".to_owned(), ..config.clone() };
                DynamicIndex::build(init, &vector_ids[..num_vectors_init], &float32, None, sync_kind)?
            } else {
                DynamicIndex::build(init, &vector_ids[..num_vectors_init], &config, training, sync_kind)?
            };
            let mut index_build_time = start.elapsed().as_secs_f64();
            logger.info(json!({"index_build_time": index_build_time}));
            if convert_init {
                let start = Instant::now();
                index = index.convert(&config, training, sync_kind)?;
                let index_convert_time = start.elapsed().as_secs_f64();
                logger.info(json!({"index_convert_time": index_convert_time}));
                index_build_time += index_convert_time;
            }
            ingest_time[0] = index_build_time;
            index
        }
        Some(idx_dir) => DynamicIndex::load(idx_dir, dim, &config, sync_kind)?,
    };
    logger.info(json!({"sync_kind": index.sync_kind()?.name()}));

    if pool_threads != max_threads_init {
        index.set_num_threads(pool_threads)?;
    }
    for batch_idx in 0..num_batches {
        let init_batch = batch_idx * args.batch_size + num_vectors_init;
        let end_batch = (init_batch + args.batch_size).min(num_vectors);
        let batch_vectors = &vectors[init_batch * dim..end_batch * dim];
        let batch_ids = &vector_ids[init_batch..end_batch];
        let start = Instant::now();
        utils::run_in_threads(max_threads, end_batch - init_batch, args.call_size, |first, count| {
            index.add(&batch_vectors[first * dim..(first + count) * dim], &batch_ids[first..first + count])
        })?;
        ingest_time[batch_idx + 1] = start.elapsed().as_secs_f64();
        eprint!("\rbatch {}/{num_batches}", batch_idx + 1);
    }
    if num_batches > 0 {
        eprintln!();
        let add_time = &ingest_time[1..];
        let add_time_mean = utils::mean(add_time);
        logger.info(json!({"add_time": {
            "mean": add_time_mean,
            "rsd": utils::std_ddof(add_time) / add_time_mean,
        }}));
        logger.info(json!({"build_and_add_time": ingest_time.iter().sum::<f64>()}));
    }
    if args.convert_after_add && args.idx_dir.is_none() {
        let start = Instant::now();
        index = index.convert(&config, training, sync_kind)?;
        logger.info(json!({"index_convert_time": start.elapsed().as_secs_f64()}));
        logger.info(json!({"sync_kind": index.sync_kind()?.name()}));
    }
    logger.info(json!({"index_size": index.size()?}));

    let name = [
        "svs_capi".to_owned(),
        format!("vecs_file={}", args.vecs_file.file_name().unwrap().to_string_lossy()),
        format!("distance={}", args.distance),
        format!("num_vectors={num_vectors}"),
        format!("graph_max_degree={}", args.graph_max_degree),
        format!("window_size={}", args.window_size),
        format!("num_vectors_init={num_vectors_init}"),
        format!("batch_size={}", args.batch_size),
        format!("svs_type={}", args.svs_type),
        "shuffle=False".to_owned(),
        format!("idx_dir={}", if args.idx_dir.is_some() { "True" } else { "False" }),
    ]
    .join("__");
    utils::save_npy(&args.out_dir.join(format!("{name}.ingest.npy")), &ingest_time)?;
    if !args.no_save {
        let idx_dir_out = args.out_dir.join(&name);
        index.save(&idx_dir_out)?;
        logger.info(json!({"index_saved": idx_dir_out}));
    }
    Ok(())
}
