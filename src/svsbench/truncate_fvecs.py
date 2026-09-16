# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Copy the first ``num_vectors`` vectors of an ``fvecs`` file into a new ``fvecs`` file.

The input file is never loaded in its entirety: vectors are streamed through a small,
fixed-size buffer, so memory use is independent of the file size.

An ``fvecs`` file is a flat sequence of records, one per vector::

    [uint32 little-endian dimensionality][dimensionality * float32 little-endian]

Since every vector in a dataset has the same dimensionality, all records have the same
size and the element type can be inferred from the file size alone (see ``read_vecs`` in
SVS ``bindings/python/src/svs/common.py`` for the reading side of the format).

Example:

    python truncate_fvecs.py base.fvecs 10000 base_first10k.fvecs
"""

import argparse
import os
import struct
from warnings import warn

# Size of the per-vector dimensionality header, in bytes.
HEADER_SIZE = 4

# Size of a float32 element, in bytes. This is what we are checking for.
FLOAT32_SIZE = 4

# Target size of the streaming buffer. The actual buffer holds a whole number of vectors,
# so it may be larger than this for very high-dimensional data.
TARGET_BUFFER_BYTES = 8 * 1024 * 1024


def check_float32(filename: str) -> tuple[int, int]:
    """
    Check that ``filename`` holds float32 vectors.

    Args:
        filename: The ``fvecs`` file to check.

    Returns:
        The dimensionality from the first two vectors.
        The number of vectors in the file, inferred from the file size and the dimensionality.

    Raises:
        ValueError: If the dimensionalities are not consistent.
    """

    file_size = os.path.getsize(filename)
    if file_size == 0:
        raise ValueError(f"{filename} is empty.")

    first_header = struct.unpack("<I", open(filename, "rb").read(HEADER_SIZE))[0]
    record_size = HEADER_SIZE + first_header * FLOAT32_SIZE
    total_vectors, remainder = divmod(file_size, record_size)
    if remainder != 0:
        warn(
            f"{filename} has {file_size} bytes, which is not divisible by "
            f"record_size = {record_size}. The file may be corrupted or incomplete."
        )

    payload_size = record_size - HEADER_SIZE
    if payload_size <= 0:
        raise ValueError(
            f"{filename} has only {record_size} bytes per vector, which leaves no room "
            f"for vector data after the {HEADER_SIZE}-byte dimensionality header."
        )

    dimensions, remainder = divmod(payload_size, FLOAT32_SIZE)
    if remainder != 0:
        raise ValueError(
            f"{filename} has {payload_size} data bytes per vector, which is not "
            f"divisible by {FLOAT32_SIZE}. The file is not encoded using float32."
        )

    return dimensions, total_vectors


def truncate_fvecs(
    input_filename: str,
    output_filename: str,
    num_vectors: int,
    dimensions: int,
) -> None:
    """
    Stream the first ``num_vectors`` vectors of ``input_filename`` into
    ``output_filename``, reading at most ``TARGET_BUFFER_BYTES`` at a time.

    The dimensionality header of every copied vector is validated against
    ``dimensions``.
    """

    record_size = HEADER_SIZE + dimensions * FLOAT32_SIZE
    batch_size = max(1, TARGET_BUFFER_BYTES // record_size)

    with open(input_filename, "rb") as fin, open(output_filename, "wb") as fout:
        remaining = num_vectors
        index = 0
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            buffer = fin.read(this_batch * record_size)
            if len(buffer) != this_batch * record_size:
                raise ValueError(
                    f"{input_filename} ended after {index} vectors, "
                    f"expected at least {num_vectors}."
                )

            for offset in range(0, len(buffer), record_size):
                encoded = struct.unpack_from("<I", buffer, offset)[0]
                if encoded != dimensions:
                    raise ValueError(
                        f"Vector {index + offset // record_size} of {input_filename} "
                        f"declares {encoded} dimensions, expected {dimensions}."
                    )

            fout.write(buffer)
            index += this_batch
            remaining -= this_batch


def main(argv: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="The fvecs file to read.")
    parser.add_argument(
        "num_vectors",
        type=int,
        help="The number of leading vectors to copy to the output file.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="The fvecs file to write. Defaults to the input file name with a "
        "'_first{num_vectors}' suffix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args(argv)

    if not args.input.endswith(".fvecs"):
        parser.error(f"Expected a '.fvecs' input file, got {args.input}.")

    stem, extension = os.path.splitext(args.input)
    output = args.output or f"{stem}_first{args.num_vectors}{extension}"

    if os.path.exists(output) and not args.force:
        parser.error(f"{output} already exists. Pass --force to overwrite it.")
    if os.path.exists(output) and os.path.samefile(args.input, output):
        parser.error("The input and output files must be different.")

    try:
        dimensions, total_vectors = check_float32(args.input)
    except ValueError as error:
        parser.error(str(error))

    if args.num_vectors <= 0:
        parser.error(f"num_vectors must be positive, got {args.num_vectors}.")
    if args.num_vectors > total_vectors:
        parser.error(
            f"num_vectors = {args.num_vectors} exceeds the "
            f"{total_vectors} vectors in {args.input}."
        )

    print(
        f"{args.input}: {total_vectors} float32 vectors of {dimensions} dimensions."
    )
    print(f"Writing the first {args.num_vectors} vectors to {output}.")

    try:
        truncate_fvecs(args.input, output, args.num_vectors, dimensions)
    except ValueError as error:
        os.remove(output)
        parser.error(str(error))

    print(f"Wrote {os.path.getsize(output)} bytes.")


if __name__ == "__main__":
    main()
