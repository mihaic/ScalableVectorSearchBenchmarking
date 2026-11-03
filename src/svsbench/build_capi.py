# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark SVS dynamic index addition using the SVS C API.

The index uses SVS_SYNC_KIND_FINE_GRAIN. Vectors are added by
max_threads threads concurrently, call_size vectors per C API call.
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

logger = logging.getLogger(__file__)


def main(
    vecs_file: Annotated[
        Path, typer.Option("--vecs_file", help="Vectors *vecs file")
    ],
    out_dir: Annotated[
        Path, typer.Option("--out_dir", help="Output dir")
    ] = Path("out"),
    log_dir: Annotated[
        Path | None, typer.Option("--log_dir", help="Log dir")
    ] = None,
    idx_dir: Annotated[
        Path | None,
        typer.Option(
            "--idx_dir", help="Index dir to load instead of building"
        ),
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
    num_vectors: Annotated[
        int | None, typer.Option("--num_vectors", help="Number of vectors")
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch_size", help="Number of vectors added per timed batch"
        ),
    ] = 10000,
    num_vectors_init: Annotated[
        int | None,
        typer.Option(
            "--num_vectors_init", help="Number of vectors to add initially"
        ),
    ] = None,
    proportion_vectors_init: Annotated[
        float | None,
        typer.Option(
            "--proportion_vectors_init",
            help="Proportion of vectors to add initially",
        ),
    ] = None,
    num_vectors_delete: Annotated[
        int,
        typer.Option(
            "--num_vectors_delete",
            help="Number of vectors to delete after each batch",
        ),
    ] = 0,
    graph_max_degree: Annotated[int, typer.Option("--graph_max_degree")] = 64,
    window_size: Annotated[int, typer.Option("--window_size")] = 200,
    alpha: Annotated[float | None, typer.Option("--alpha")] = None,
    max_threads: Annotated[
        int,
        typer.Option(
            "--max_threads", help="Number of threads calling the C API"
        ),
    ] = utils.default_max_threads(),
    max_threads_init: Annotated[
        int | None,
        typer.Option(
            "--max_threads_init",
            help="Number of index threads for building the initial graph"
            " [default: max_threads]",
        ),
    ] = None,
    pool_threads: Annotated[
        int | None,
        typer.Option(
            "--pool_threads",
            help="Number of index threads after the initial build."
            " With SVS_SYNC_KIND_FINE_GRAIN, every add runs on an index"
            " thread, so this limits concurrency [default: max_threads]",
        ),
    ] = None,
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
            help="Number of vectors or IDs per C API call."
            " SVS_SYNC_KIND_FINE_GRAIN may require 1",
        ),
    ] = 1,
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle",
            help="Shuffle order of vectors. Either search with"
            " svsbench.search_capi --shuffle or regenerate the ground"
            " truth with svsbench.generate_ground_truth --shuffle",
        ),
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    leanvec_dims: Annotated[
        int,
        typer.Option(
            "--leanvec_dims",
            help="LeanVec dimensionality (negative: divisor of dimensionality)",
        ),
    ] = consts.DEFAULT_LEANVEC_DIMS,
    train_query_file: Annotated[
        Path | None,
        typer.Option(
            "--train_query_file",
            help="Query *vecs file for LeanVec out-of-distribution training",
        ),
    ] = None,
    train_max_vectors: Annotated[
        int,
        typer.Option(
            "--train_max_vectors",
            help="Maximum number of base vectors from vecs_file"
            " to use for LeanVec out-of-distribution training (0 for all)",
        ),
    ] = consts.DEFAULT_LEANVEC_TRAIN_MAX_VECTORS,
    convert: Annotated[
        bool,
        typer.Option(
            "--convert",
            help="Build the initial index with float32 and convert it to"
            " svs_type with svs_index_convert_dynamic_ex",
        ),
    ] = False,
    no_save: Annotated[
        bool, typer.Option("--no_save", help="Do not save built index")
    ] = False,
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
    out_dir.mkdir(exist_ok=True)
    index, name, ingest_time, delete_time = build_dynamic(
        vecs_path=vecs_file,
        svs_type=svs_type,
        distance=distance,
        idx_dir=idx_dir,
        num_vectors=num_vectors,
        graph_max_degree=graph_max_degree,
        window_size=window_size,
        alpha=alpha,
        max_threads=max_threads,
        max_threads_init=max_threads_init,
        pool_threads=pool_threads,
        threadpool=threadpool,
        call_size=call_size,
        batch_size=batch_size,
        num_vectors_delete=num_vectors_delete,
        num_vectors_init=num_vectors_init,
        proportion_vectors_init=proportion_vectors_init,
        shuffle=shuffle,
        seed=seed,
        leanvec_dims=leanvec_dims,
        train_query_path=train_query_file,
        train_max_vectors=train_max_vectors,
        convert=convert,
    )
    with index:
        np.save(out_dir / (name + ".ingest.npy"), ingest_time)
        if num_vectors_delete > 0:
            np.save(out_dir / (name + ".delete.npy"), delete_time)
        if not no_save:
            idx_dir_out = out_dir / name
            index.save(idx_dir_out)
            logger.info({"index_saved": idx_dir_out})


def read_vectors(
    vecs_path: Path, num_vectors: int | None = None
) -> npt.NDArray:
    """Read vectors as a contiguous float32 array."""
    if vecs_path.suffix not in consts.SUFFIX_TO_SVS_TYPE:
        raise ValueError("Unknown suffix: " + vecs_path.suffix)
    vectors = merge.read_vecs(vecs_path)
    if num_vectors is not None and num_vectors > vectors.shape[0]:
        raise ValueError(
            f"{num_vectors=} is greater than the number of vectors"
            f" in {vecs_path=!s}, {vectors.shape[0]}"
        )
    return np.ascontiguousarray(vectors[:num_vectors], dtype=np.float32)


def build_dynamic(
    *,
    vecs_path: Path,
    svs_type: str,
    distance: str,
    idx_dir: Path | None = None,
    num_vectors: int | None = None,
    graph_max_degree: int = 64,
    window_size: int = 200,
    alpha: float | None = None,
    max_threads: int = 1,
    max_threads_init: int | None = None,
    pool_threads: int | None = None,
    threadpool: str = "native",
    call_size: int = 1,
    batch_size: int = 10000,
    num_vectors_delete: int = 0,
    num_vectors_init: int | None = None,
    proportion_vectors_init: float | None = None,
    shuffle: bool = False,
    seed: int = 42,
    leanvec_dims: int = consts.DEFAULT_LEANVEC_DIMS,
    train_query_path: Path | None = None,
    train_max_vectors: int = consts.DEFAULT_LEANVEC_TRAIN_MAX_VECTORS,
    convert: bool = False,
) -> tuple[capi.DynamicIndex, str, npt.NDArray, npt.NDArray | None]:
    """Build SVS index with the C API."""
    logger.info({"build_args": locals()})
    logger.info(utils.read_system_config())
    logger.info(capi.runtime_info())

    vectors = read_vectors(vecs_path, num_vectors)
    num_vectors = vectors.shape[0]
    if num_vectors_init is None:
        if proportion_vectors_init is None:
            num_vectors_init = batch_size
        else:
            num_vectors_init = int(num_vectors * proportion_vectors_init)
    if max_threads_init is None:
        max_threads_init = max_threads
    if pool_threads is None:
        pool_threads = max_threads
    if train_query_path is not None:
        train_vectors = vectors[: train_max_vectors or None]
        train_queries = read_vectors(train_query_path)
    else:
        train_vectors = train_queries = None
    if shuffle:
        vectors = vectors[np.random.default_rng(seed).permutation(num_vectors)]
    alpha = consts.STR_TO_ALPHA[distance] if alpha is None else alpha

    num_batches = int(np.ceil((num_vectors - num_vectors_init) / batch_size))
    ingest_time = np.zeros(num_batches + 1)
    delete_time = np.zeros(num_batches + 1) if num_vectors_delete > 0 else None
    vector_ids = np.arange(num_vectors, dtype=np.uintp)
    present = np.zeros(num_vectors, dtype=bool)

    if idx_dir is None:
        builder_kwargs = {
            "distance": distance,
            "graph_max_degree": graph_max_degree,
            "window_size": window_size,
            "alpha": alpha,
            "num_threads": max_threads_init,
            "threadpool": threadpool,
            "leanvec_dims": leanvec_dims,
        }
        storage_kwargs = {
            "svs_type": svs_type,
            "train_vectors": train_vectors,
            "train_queries": train_queries,
        }
        start = time.perf_counter()
        index = capi.DynamicIndex.build(
            vectors[:num_vectors_init],
            vector_ids[:num_vectors_init],
            **builder_kwargs,
            **({"svs_type": "float32"} if convert else storage_kwargs),
        )
        index_build_time = time.perf_counter() - start
        logger.info({"index_build_time": index_build_time})
        if convert:
            start = time.perf_counter()
            with index:
                index = index.convert(**builder_kwargs, **storage_kwargs)
            index_convert_time = time.perf_counter() - start
            logger.info({"index_convert_time": index_convert_time})
            index_build_time += index_convert_time
        ingest_time[0] = index_build_time
    else:
        index = capi.DynamicIndex.load(
            idx_dir,
            dim=vectors.shape[1],
            svs_type=svs_type,
            distance=distance,
            num_threads=max_threads_init,
            threadpool=threadpool,
            leanvec_dims=leanvec_dims,
        )
    present[:num_vectors_init] = True
    logger.info({"sync_kind": index.sync_kind.name})

    rng_delete = np.random.default_rng(seed)
    if pool_threads != max_threads_init:
        index.num_threads = pool_threads
    vector_bytes = vectors.strides[0]
    id_bytes = vector_ids.strides[0]
    with ThreadPoolExecutor(max_threads) as executor:
        for batch_idx in tqdm(range(num_batches)):
            init_batch = batch_idx * batch_size + num_vectors_init
            end_batch = min(init_batch + batch_size, num_vectors)
            vectors_address = vectors[init_batch:].ctypes.data
            ids_address = vector_ids[init_batch:].ctypes.data

            def add(first: int, count: int) -> None:
                index.add_address(
                    vectors_address + first * vector_bytes,
                    ids_address + first * id_bytes,
                    count,
                )

            start = time.perf_counter()
            utils.run_in_threads(
                executor, max_threads, end_batch - init_batch, call_size, add
            )
            ingest_time[batch_idx + 1] = time.perf_counter() - start
            present[init_batch:end_batch] = True
            if num_vectors_delete > 0:
                ids_to_delete = np.ascontiguousarray(
                    rng_delete.choice(
                        np.flatnonzero(present),
                        size=num_vectors_delete,
                        replace=False,
                    ),
                    dtype=np.uintp,
                )
                delete_address = ids_to_delete.ctypes.data

                def delete(first: int, count: int) -> None:
                    index.delete_address(
                        delete_address + first * id_bytes, count
                    )

                start = time.perf_counter()
                utils.run_in_threads(
                    executor,
                    max_threads,
                    num_vectors_delete,
                    call_size,
                    delete,
                )
                delete_time[batch_idx + 1] = time.perf_counter() - start
                present[ids_to_delete] = False
    if num_batches:
        add_time_mean = np.mean(ingest_time[1:])
        add_time_rsd = (
            np.std(ingest_time[1:], ddof=min(1, num_batches - 1))
            / add_time_mean
        )
        logger.info({"add_time": {"mean": add_time_mean, "rsd": add_time_rsd}})
        logger.info({"build_and_add_time": np.sum(ingest_time)})
    logger.info({"index_size": index.size})

    name = "__".join(
        (
            "svs_capi",
            "vecs_file=" + vecs_path.name,
            "distance=" + distance,
            "num_vectors=" + str(num_vectors),
            "graph_max_degree=" + str(graph_max_degree),
            "window_size=" + str(window_size),
            "num_vectors_init=" + str(num_vectors_init),
            "batch_size=" + str(batch_size),
            "svs_type=" + svs_type,
            "shuffle=" + (str(seed) if shuffle else "False"),
            "idx_dir=" + str(idx_dir is not None),
        )
    )
    return index, name, ingest_time, delete_time


if __name__ == "__main__":
    # https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    typer.run(main)
