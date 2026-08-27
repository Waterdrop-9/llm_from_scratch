from pathlib import Path

import pytest

from src.train_bpe import train_bpe
from src.tokenizer import Tokenizer


def _write_corpus(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_train_bpe_initializes_bytes_and_special_tokens(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path, "a")

    vocab, merges = train_bpe(corpus_path, vocab_size=257, special_tokens=["<|end|>"])

    assert len(vocab) == 257
    assert vocab[0] == b"\x00"
    assert vocab[255] == b"\xff"
    assert vocab[256] == b"<|end|>"
    assert merges == []


def test_train_bpe_selects_highest_frequency_pair(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path, "abab abab")

    vocab, merges = train_bpe(corpus_path, vocab_size=257, special_tokens=[])

    assert merges == [(b"a", b"b")]
    assert vocab[256] == b"ab"


def test_train_bpe_uses_lexicographically_greater_pair_on_tie(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path, "ab ac")

    _, merges = train_bpe(corpus_path, vocab_size=257, special_tokens=[])

    assert merges == [(b"a", b"c")]


def test_train_bpe_does_not_merge_across_special_tokens(tmp_path: Path):
    special = "<|end|>"
    corpus_path = _write_corpus(tmp_path, f"ab{special}cd")

    vocab, merges = train_bpe(corpus_path, vocab_size=258, special_tokens=[special])

    assert vocab[256] == special.encode("utf-8")
    assert all(special.encode("utf-8") not in pair for pair in merges)
    assert all(token != b"b<" and token != b">c" for token in vocab.values())


def test_trained_vocab_and_merges_work_with_tokenizer(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path, "hello hello")
    vocab, merges = train_bpe(corpus_path, vocab_size=260, special_tokens=[])
    tokenizer = Tokenizer(vocab, merges)

    ids = tokenizer.encode("hello")

    assert ids
    assert tokenizer.decode(ids) == "hello"


def test_train_bpe_rejects_vocab_smaller_than_initial_vocabulary(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path, "hello")

    with pytest.raises(ValueError, match="at least 256"):
        train_bpe(corpus_path, vocab_size=255, special_tokens=[])
