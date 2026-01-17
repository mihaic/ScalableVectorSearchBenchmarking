"""Convert data between formats."""
from pathlib import Path

import h5py
import numpy as np
import svs
import typer

from . import consts

app = typer.Typer(help=__doc__)

@app.command()
def hdf5tovecs(input_path: Path, output_dir: Path = Path(".")):
    with h5py.File(input_path) as file:
        name_prefix = output_dir / input_path.stem
        neighbors = np.array(file["neighbors"])
        if neighbors.dtype.kind not in "iu":
            raise ValueError(f"Neighbors dtype not integer: {neighbors.dtype}")
        if np.any(neighbors < 0):
            raise ValueError("Negative neighbors found")
        if np.any(neighbors > np.iinfo(np.uint32).max):
            raise ValueError("Neighbors exceed uint32 max value")
        test = np.array(file["test"])
        if test.dtype == np.float64:
            test = test.astype(np.float32)
        train = np.array(file["train"])
        if train.dtype == np.float64:
            train = train.astype(np.float32)
        svs.write_vecs(
            neighbors.astype(np.uint32), f"{name_prefix}_neighbors.ivecs"
        )
        svs.write_vecs(
            test,
            f"{name_prefix}_test{consts.DTYPE_TO_SUFFIX[test.dtype.type]}",
        )
        svs.write_vecs(
            train,
            f"{name_prefix}_train{consts.DTYPE_TO_SUFFIX[train.dtype.type]}",
        )

@app.command()
def vecstohdf5(train_path: Path, query_path: Path, ground_truth_path: Path, output_path: Path):
    with h5py.File(output_path, "w") as file:
        train = svs.read_vecs(str(train_path))
        query = svs.read_vecs(str(query_path))
        ground_truth = svs.read_vecs(str(ground_truth_path))
        file.create_dataset("train", data=train)
        file.create_dataset("test", data=query)
        file.create_dataset("neighbors", data=ground_truth)

if __name__ == "__main__":
    app()
