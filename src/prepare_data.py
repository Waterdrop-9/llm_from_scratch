from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer


def tokenize_file_to_uint16(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    buffer_size: int = 1_000_000,
) -> int:
    """
    Stream a text file through the tokenizer and save token IDs as uint16.

    Returns:
        The number of token IDs written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    if output_path.exists():
        raise FileExistsError(f"Output token file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    max_uint16 = np.iinfo(np.uint16).max
    token_buffer: list[int] = []
    total_tokens = 0

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        temporary_path.open("wb") as output_file,
    ):
        for token_id in tokenizer.encode_iterable(input_file):
            if token_id < 0 or token_id > max_uint16:
                raise ValueError(
                    f"Token ID {token_id} cannot be represented as uint16."
                )

            token_buffer.append(token_id)

            if len(token_buffer) >= buffer_size:
                token_array = np.asarray(
                    token_buffer,
                    dtype=np.uint16,
                )
                token_array.tofile(output_file)

                total_tokens += len(token_buffer)
                token_buffer.clear()

        # 文件读取结束后，写入最后一批不足 buffer_size 的 token。
        if token_buffer:
            token_array = np.asarray(
                token_buffer,
                dtype=np.uint16,
            )
            token_array.tofile(output_file)
            total_tokens += len(token_buffer)

    # 只有全部编码和写入成功后，才生成正式输出文件。
    temporary_path.replace(output_path)

    return total_tokens


def load_tokenized_dataset(path: str | Path) -> np.memmap:
    """Open a raw uint16 token file without loading it entirely into memory."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Token file not found: {path}")

    file_size = path.stat().st_size
    bytes_per_token = np.dtype(np.uint16).itemsize

    if file_size == 0:
        raise ValueError("Token file is empty.")

    if file_size % bytes_per_token != 0:
        raise ValueError("Token file size is not aligned to uint16.")

    return np.memmap(path, dtype=np.uint16, mode="r")
