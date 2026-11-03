# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests of the SVS C API commands.

They do not use the SVS Python bindings, so they run on free-threaded
Python. They are skipped if the SVS C API library is not found.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from svsbench import build_capi, capi, consts, search_capi, utils

try:
    capi.lib()
except OSError as error:
    pytest.skip(str(error), allow_module_level=True)

NUM_VECTORS = 1000
DIM = 32
K = 10


def write_vecs(array: np.ndarray, path: Path) -> None:
    dims = np.full((array.shape[0], 1), array.shape[1], dtype=np.int32)
    np.hstack((dims.view(array.dtype), array)).tofile(path)


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    return np.random.default_rng(42).random((NUM_VECTORS, DIM), np.float32)


@pytest.fixture(scope="module")
def queries() -> np.ndarray:
    return np.random.default_rng(43).random((100, DIM), np.float32)


@pytest.fixture(scope="module")
def data_dir(vectors, queries, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data")
    write_vecs(vectors, path / "base.fvecs")
    write_vecs(queries, path / "query.fvecs")
    distances = ((queries[:, None, :] - vectors[None, :, :]) ** 2).sum(-1)
    ground_truth = np.argsort(distances, axis=1)[:, :100].astype(np.uint32)
    write_vecs(ground_truth, path / "gt.ivecs")
    return path


@pytest.mark.parametrize("svs_type", consts.SVS_TYPES)
def test_dynamic_index(svs_type, vectors, tmp_path):
    ids = np.arange(NUM_VECTORS, dtype=np.uintp)
    num_init = NUM_VECTORS // 2
    with capi.DynamicIndex.build(
        vectors[:num_init],
        ids[:num_init],
        svs_type=svs_type,
        distance="l2",
        graph_max_degree=16,
        window_size=32,
        num_threads=2,
    ) as index:
        assert index.sync_kind == capi.SyncKind.FINE_GRAIN
        with ThreadPoolExecutor(4) as executor:
            executor.map(
                lambda i: index.add(vectors[i : i + 1], ids[i : i + 1]),
                range(num_init, NUM_VECTORS),
            )
        assert index.size == NUM_VECTORS
        index.delete(ids[:1])
        assert index.size == NUM_VECTORS - 1
        index.save(tmp_path / "index")
    with capi.DynamicIndex.load(
        tmp_path / "index", dim=DIM, svs_type=svs_type, distance="l2"
    ) as index:
        assert index.size == NUM_VECTORS - 1
        queries = vectors[-10:]
        indices = np.zeros((len(queries), K), dtype=np.uintp)
        distances = np.zeros((len(queries), K), dtype=np.float32)
        searcher = capi.Searcher(index, K, 1, indices, distances)
        for i in range(len(queries)):
            searcher.search_address(queries[i:].ctypes.data, i, 1)
        assert (indices[:, 0] == ids[-10:]).mean() >= 0.9


def test_omp_threadpool(vectors):
    try:
        index = capi.DynamicIndex.build(
            vectors, num_threads=2, threadpool="omp"
        )
    except capi.CAPIError as error:
        if error.code == capi.ErrorCode.NOT_IMPLEMENTED:
            pytest.skip(str(error))
        raise
    with index:
        assert index.num_threads == 2


def test_error(vectors):
    with pytest.raises(capi.CAPIError, match="code"):
        capi.DynamicIndex.build(vectors, graph_max_degree=0)


def test_error_per_thread(vectors):
    errors = []

    def fail() -> None:
        try:
            capi.DynamicIndex.build(vectors, graph_max_degree=0)
        except capi.CAPIError as error:
            errors.append(error)

    threads = [threading.Thread(target=fail) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(errors) == len(threads)


@pytest.mark.parametrize("call_size", (1, 7))
@pytest.mark.parametrize("num_threads", (1, 4))
def test_run_in_threads(call_size, num_threads):
    num_items = 100
    counts = np.zeros(num_items, dtype=int)

    def function(first: int, count: int) -> None:
        assert count <= call_size
        counts[first : first + count] += 1

    with ThreadPoolExecutor(num_threads) as executor:
        utils.run_in_threads(
            executor, num_threads, num_items, call_size, function
        )
    assert (counts == 1).all()


def test_k_recall_at():
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    results = np.array([[0, 2, 9], [9, 8, 7]])
    assert utils.k_recall_at(ground_truth, results, 2, 3) == 0.25
    assert utils.k_recall_at(ground_truth, results, 3, 3) == 2 / 6


def test_k_recall_at_matches_svs():
    svs = pytest.importorskip("svs")
    rng = np.random.default_rng(42)
    ground_truth = np.stack([rng.permutation(50)[:20] for _ in range(100)])
    results = np.stack([rng.permutation(50)[:10] for _ in range(100)])
    for k, at in ((10, 10), (5, 10), (1, 1)):
        assert utils.k_recall_at(
            ground_truth, results, k, at
        ) == pytest.approx(
            svs.k_recall_at(
                ground_truth.astype(np.uint32),
                results.astype(np.uint64),
                k,
                at,
            )
        )


@pytest.mark.parametrize(
    "call_size,num_vectors_delete", ((1, 0), (16, 0), (1, 10))
)
@pytest.mark.parametrize("shuffle", (False, True))
def test_build_and_search(
    call_size, num_vectors_delete, shuffle, data_dir, tmp_path
):
    try:
        _build_and_search(
            call_size, num_vectors_delete, shuffle, data_dir, tmp_path
        )
    except capi.CAPIError as error:
        # SVS_SYNC_KIND_FINE_GRAIN may support only one item per call
        if call_size > 1 and error.code == capi.ErrorCode.NOT_IMPLEMENTED:
            pytest.skip(str(error))
        raise


def _build_and_search(
    call_size, num_vectors_delete, shuffle, data_dir, tmp_path
):
    index, name, ingest_time, delete_time = build_capi.build_dynamic(
        vecs_path=data_dir / "base.fvecs",
        svs_type="float32",
        distance="l2",
        graph_max_degree=16,
        window_size=32,
        max_threads=4,
        call_size=call_size,
        batch_size=200,
        num_vectors_init=200,
        num_vectors_delete=num_vectors_delete,
        shuffle=shuffle,
    )
    with index:
        assert len(ingest_time) == 5
        assert (delete_time is None) == (num_vectors_delete == 0)
        assert index.size == NUM_VECTORS - 4 * num_vectors_delete
        index.save(tmp_path / name)
    results, _, recall = search_capi.search(
        idx_dir=tmp_path / name,
        query_path=data_dir / "query.fvecs",
        ground_truth_path=data_dir / "gt.ivecs",
        search_window_sizes=[10, 50],
        distance="l2",
        count=K,
        max_threads=4,
        call_size=call_size,
        num_rep=2,
        shuffle=shuffle,
        num_vectors=NUM_VECTORS,
    )
    assert results.shape == (100, K)
    if num_vectors_delete == 0:
        assert recall > 0.8


def test_main(data_dir, tmp_path):
    runner = CliRunner()
    build_app = typer.Typer()
    build_app.command()(build_capi.main)
    result = runner.invoke(
        build_app,
        [
            "--vecs_file",
            str(data_dir / "base.fvecs"),
            "--svs_type",
            "leanvec4x8",
            "--train_query_file",
            str(data_dir / "query.fvecs"),
            "--distance",
            "l2",
            "--graph_max_degree",
            "16",
            "--window_size",
            "32",
            "--batch_size",
            "500",
            "--max_threads",
            "2",
            "--out_dir",
            str(tmp_path),
            utils.UNCOMMITTED,
        ],
    )
    assert result.exit_code == 0, result.output
    (idx_dir,) = (path for path in tmp_path.iterdir() if path.is_dir())
    assert (tmp_path / (idx_dir.name + ".ingest.npy")).is_file()
    search_app = typer.Typer()
    search_app.command()(search_capi.main)
    result = runner.invoke(
        search_app,
        [
            "--idx_dir",
            str(idx_dir),
            "--svs_type",
            "leanvec4x8",
            "--query_file",
            str(data_dir / "query.fvecs"),
            "--ground_truth_file",
            str(data_dir / "gt.ivecs"),
            "--search_window_size",
            "20",
            "--distance",
            "l2",
            "--max_threads",
            "2",
            "--num_rep",
            "1",
            "--out_dir",
            str(tmp_path),
            utils.UNCOMMITTED,
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("ood", (False, True))
@pytest.mark.parametrize("convert", (False, True))
def test_build_and_search_leanvec(convert, ood, data_dir, tmp_path):
    base_path = data_dir / "base.fvecs"
    index, name, ingest_time, _ = build_capi.build_dynamic(
        vecs_path=base_path,
        svs_type="leanvec4x8",
        distance="l2",
        graph_max_degree=16,
        window_size=32,
        max_threads=4,
        batch_size=200,
        num_vectors_init=200,
        leanvec_dims=16,
        train_query_path=base_path if ood else None,
        convert=convert,
    )
    with index:
        assert index.size == NUM_VECTORS
        index.save(tmp_path / name)
    _, _, recall = search_capi.search(
        idx_dir=tmp_path / name,
        query_path=data_dir / "query.fvecs",
        ground_truth_path=data_dir / "gt.ivecs",
        search_window_sizes=[50],
        svs_type="leanvec4x8",
        distance="l2",
        count=K,
        max_threads=4,
        num_rep=1,
        leanvec_dims=16,
    )
    # Uniform random data has no low-dimensional structure for LeanVec
    assert recall > 0.5
