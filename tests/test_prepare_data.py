import numpy as np
import pytest

from src.data_loader import get_batch
from src.prepare_data import load_tokenized_dataset, tokenize_file_to_uint16
from src.tokenizer import Tokenizer


def byte_tokenizer() -> Tokenizer:
    return Tokenizer(
        vocab={token_id: bytes([token_id]) for token_id in range(256)},
        merges=[],
    )


def test_tokenize_file_writes_every_token_across_buffer_boundaries(tmp_path):
    input_path = tmp_path / "stories.txt"
    output_path = tmp_path / "stories.bin"
    input_path.write_text("hello\nworld", encoding="utf-8")

    token_count = tokenize_file_to_uint16(
        byte_tokenizer(),
        input_path,
        output_path,
        buffer_size=3,
    )

    expected = list("hello\nworld".encode("utf-8"))
    actual = np.fromfile(output_path, dtype=np.uint16)

    assert token_count == len(expected)
    assert actual.tolist() == expected
    assert not output_path.with_suffix(".bin.tmp").exists()


def test_load_tokenized_dataset_returns_read_only_memmap(tmp_path):
    path = tmp_path / "tokens.bin"
    np.asarray([4, 8, 15, 16, 23, 42], dtype=np.uint16).tofile(path)

    dataset = load_tokenized_dataset(path)

    assert isinstance(dataset, np.memmap)
    assert dataset.dtype == np.uint16
    assert dataset.shape == (6,)
    assert dataset.tolist() == [4, 8, 15, 16, 23, 42]
    assert not dataset.flags.writeable


def test_memmap_can_feed_get_batch(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    dataset = load_tokenized_dataset(path)

    inputs, targets = get_batch(
        dataset,
        batch_size=4,
        context_length=5,
        device="cpu",
    )

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert (targets == inputs + 1).all()


def test_tokenize_file_rejects_token_id_outside_uint16(tmp_path):
    input_path = tmp_path / "stories.txt"
    output_path = tmp_path / "stories.bin"
    input_path.write_text("a", encoding="utf-8")
    tokenizer = Tokenizer(vocab={65_536: b"a"}, merges=[])

    with pytest.raises(ValueError, match="cannot be represented as uint16"):
        tokenize_file_to_uint16(tokenizer, input_path, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize("contents", [b"", b"\x01"])
def test_load_tokenized_dataset_rejects_invalid_files(tmp_path, contents):
    path = tmp_path / "tokens.bin"
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        load_tokenized_dataset(path)
