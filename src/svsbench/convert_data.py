from pathlib import Path

import h5py
import numpy as np
import svs
import typer

from . import consts


def convert_hdf5(input_path: Path, output_dir: Path):
    with h5py.File(input_path) as file:
        name_prefix = output_dir / input_path.stem
        test = np.array(file["test"])
        svs.write_vecs(
            test,
            f"{name_prefix}_test{consts.DTYPE_TO_SUFFIX[test.dtype.type]}",
        )
        train = np.array(file["train"])
        svs.write_vecs(
            train,
            f"{name_prefix}_train{consts.DTYPE_TO_SUFFIX[train.dtype.type]}",
        )
        neighbors = np.array(file["neighbors"])
        if np.any(neighbors < 0):
            raise ValueError("Negative neighbors found")
        if neighbors.dtype.type is not np.int32:
            raise ValueError("Neighbors dataset is not int32")
        svs.write_vecs(
            neighbors.astype(np.uint32),
            f"{name_prefix}_neighbors.ivecs",
        )


def main(input_path: Path, output_dir: Path = Path(".")):
    match suffix := input_path.suffix:
        case ".hdf5":
            convert_hdf5(input_path, output_dir)
        case _:
            raise ValueError(f"Unsupported suffix: {suffix}")


if __name__ == "__main__":
    typer.run(main)
