# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Search benchmark using the SVS C API.

The index is loaded with SVS_SYNC_KIND_FINE_GRAIN. Queries are searched
by max_threads threads concurrently, call_size queries per C API call.
Does not use the SVS Python bindings, so it runs on free-threaded Python.
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import click
import numpy as np
import numpy.typing as npt
import typer
from tqdm import tqdm

from . import capi, consts, merge, utils
from .build_capi import read_vectors

logger = logging.getLogger(__file__)


def main(
    idx_dir: Annotated[Path, typer.Option("--idx_dir", help="Index dir")],
    query_file: Annotated[
        Path, typer.Option("--query_file", help="Query *vecs file")
    ],
    ground_truth_file: Annotated[
        Path,
        typer.Option("--ground_truth_file", help="Ground truth ivecs file"),
    ],
    search_window_size: Annotated[
        list[int],
        typer.Option(
            "--search_window_size",
            help="Search window size (repeat to search with several)",
        ),
    ],
    out_dir: Annotated[
        Path, typer.Option("--out_dir", help="Output dir")
    ] = Path("out"),
    log_dir: Annotated[
        Path | None, typer.Option("--log_dir", help="Log dir")
    ] = None,
    svs_type: Annotated[
        str,
        typer.Option(
            "--svs_type",
            help="SVS type",
            click_type=click.Choice(consts.SVS_TYPES),
        ),
    ] = "float32",
    distance: Annotated[
        str,
        typer.Option("--distance", click_type=click.Choice(consts.DISTANCES)),
    ] = "mip",
    k: Annotated[
        int, typer.Option("-k", help="Number of neighbors to return")
    ] = 10,
    max_threads: Annotated[
        int,
        typer.Option(
            "--max_threads", help="Number of threads calling the C API"
        ),
    ] = utils.default_max_threads(),
    pool_threads: Annotated[
        int,
        typer.Option(
            "--pool_threads",
            help="Number of index threads. With 1, each search runs"
            " on its calling thread",
        ),
    ] = 1,
    threadpool: Annotated[
        str,
        typer.Option(
            "--threadpool",
            help="Index thread pool kind (omp requires a library"
            " built with OpenMP)",
            click_type=click.Choice(capi.THREADPOOLS),
        ),
    ] = "native",
    call_size: Annotated[
        int,
        typer.Option(
            "--call_size",
            help="Number of queries per C API call."
            " SVS_SYNC_KIND_FINE_GRAIN may require 1",
        ),
    ] = 1,
    num_rep: Annotated[
        int,
        typer.Option("--num_rep", help="Number of search repetitions"),
    ] = 5,
    leanvec_dims: Annotated[
        int,
        typer.Option(
            "--leanvec_dims",
            help="LeanVec dimensionality (negative: divisor of dimensionality)",
        ),
    ] = consts.DEFAULT_LEANVEC_DIMS,
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle",
            help="Search an index built with svsbench.build_capi --shuffle",
        ),
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    num_vectors: Annotated[
        int | None,
        typer.Option(
            "--num_vectors",
            help="Number of vectors the index was built with, for --shuffle"
            " [default: index size, which is wrong after deletions]",
        ),
    ] = None,
    uncommitted: Annotated[
        bool,
        typer.Option(
            utils.UNCOMMITTED,
            help="Run even if there are changes in SVSBench"
            " not committed to git",
        ),
    ] = False,
) -> None:
    log_file = utils.configure_logger(
        logger, log_dir if log_dir is not None else out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": sys.argv})
    utils.check_uncommitted_and_log_version(logger, uncommitted)
    search(
        idx_dir=idx_dir,
        query_path=query_file,
        ground_truth_path=ground_truth_file,
        search_window_sizes=search_window_size,
        svs_type=svs_type,
        distance=distance,
        count=k,
        max_threads=max_threads,
        pool_threads=pool_threads,
        threadpool=threadpool,
        call_size=call_size,
        num_rep=num_rep,
        leanvec_dims=leanvec_dims,
        shuffle=shuffle,
        seed=seed,
        num_vectors=num_vectors,
    )


def search(
    *,
    idx_dir: Path,
    query_path: Path,
    ground_truth_path: Path,
    search_window_sizes: list[int],
    svs_type: str = "float32",
    distance: str = "mip",
    count: int = 10,
    max_threads: int = 1,
    pool_threads: int = 1,
    threadpool: str = "native",
    call_size: int = 1,
    num_rep: int = 5,
    leanvec_dims: int = consts.DEFAULT_LEANVEC_DIMS,
    shuffle: bool = False,
    seed: int = 42,
    num_vectors: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """Search with each search window size and log results.

    Returns the results of the last search window size.
    """
    logger.info({"search_args": locals()})
    logger.info(utils.read_system_config())
    logger.info(capi.runtime_info())
    query = read_vectors(query_path)
    ground_truth = merge.read_vecs(ground_truth_path)
    if ground_truth.shape[1] < count:
        raise ValueError(
            f"Ground truth has {ground_truth.shape[1]} neighbors, need {count}"
        )
    num_queries = query.shape[0]
    query_bytes = query.strides[0]

    with (
        capi.DynamicIndex.load(
            idx_dir,
            dim=query.shape[1],
            svs_type=svs_type,
            distance=distance,
            num_threads=pool_threads,
            threadpool=threadpool,
            leanvec_dims=leanvec_dims,
        ) as index,
        ThreadPoolExecutor(max_threads) as executor,
    ):
        logger.info({"sync_kind": index.sync_kind.name})
        if shuffle:
            dtype = ground_truth.dtype
            permutation = np.random.default_rng(seed).permutation(
                index.size if num_vectors is None else num_vectors
            )
            ground_truth = np.argsort(permutation)[ground_truth].astype(
                dtype=dtype, casting="same_value"
            )
        results = np.zeros((num_queries, count), dtype=np.uintp)
        distances = np.zeros((num_queries, count), dtype=np.float32)
        latencies = np.zeros(-(-num_queries // call_size))
        for search_window_size in search_window_sizes:
            with capi.SearchParams(search_window_size) as search_params:
                searchers = [
                    capi.Searcher(
                        index,
                        count,
                        call_size,
                        results,
                        distances,
                        search_params,
                    )
                    for _ in range(max_threads)
                ]
                # run_in_threads assigns call i to thread i % max_threads,
                # so each searcher is used by one thread.
                searcher_of_call = [
                    searchers[call_idx % len(searchers)]
                    for call_idx in range(len(latencies))
                ]

                def search_queries(first: int, num: int) -> None:
                    call_idx = first // call_size
                    queries_address = query.ctypes.data + first * query_bytes
                    start = time.perf_counter()
                    searcher_of_call[call_idx].search_address(
                        queries_address, first, num
                    )
                    latencies[call_idx] = time.perf_counter() - start

                # Warm-up search
                utils.run_in_threads(
                    executor,
                    max_threads,
                    num_queries,
                    call_size,
                    search_queries,
                )

                qps = []
                p95s = []
                p99s = []
                mean_latencies = []
                for _ in tqdm(range(num_rep)):
                    start = time.perf_counter()
                    utils.run_in_threads(
                        executor,
                        max_threads,
                        num_queries,
                        call_size,
                        search_queries,
                    )
                    total_time = time.perf_counter() - start
                    qps.append(num_queries / total_time)
                    p95s.append(np.percentile(latencies, 95))
                    p99s.append(np.percentile(latencies, 99))
                    mean_latencies.append(np.mean(latencies))
            recall = utils.k_recall_at(ground_truth, results, count, count)
            logger.info(
                {
                    "search_results": {
                        "qps": qps,
                        "qps_mean": np.mean(qps),
                        "qps_rsd": np.std(qps, ddof=min(1, num_rep - 1))
                        / np.mean(qps),
                        "p95": p95s,
                        "p95_mean": np.mean(p95s),
                        "p95_rsd": np.std(p95s, ddof=min(1, num_rep - 1))
                        / np.mean(p95s),
                        "p99_mean": np.mean(p99s),
                        "latency_mean": np.mean(mean_latencies),
                        "search_parameters": {
                            "search_window_size": search_window_size,
                        },
                        "batch_size": call_size,
                        "max_threads": max_threads,
                        "recall": recall,
                    },
                }
            )
    return results, distances, recall


if __name__ == "__main__":
    # https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    typer.run(main)
