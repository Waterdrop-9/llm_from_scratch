import pytest

import src.tokenizer as tokenizer_module


def test_tokenizer_builds_byte_lookup_and_merge_rank():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"ab",
    }
    merges = [(b"a", b"b")]

    tokenizer = tokenizer_module.Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
    )

    assert tokenizer.vocab is vocab
    assert tokenizer.byte_to_id[b"a"] == 0
    assert tokenizer.byte_to_id[b"ab"] == 2
    assert tokenizer.merge_to_rank == {(b"a", b"b"): 0}
    assert tokenizer.special_tokens == [b"<|endoftext|>"]


def test_decode_concatenates_vocab_bytes_and_preserves_utf8():
    vocab = {
        0: "你".encode("utf-8"),
        1: "好".encode("utf-8"),
        2: b"!",
    }
    tokenizer = tokenizer_module.Tokenizer(vocab=vocab, merges=[])

    assert tokenizer.decode([0, 1, 2]) == "你好!"
    assert tokenizer.decode([]) == ""


def test_decode_rejects_unknown_token_id():
    tokenizer = tokenizer_module.Tokenizer(vocab={0: b"a"}, merges=[])

    with pytest.raises(ValueError, match="ID 7"):
        tokenizer.decode([7])


def test_encode_maps_single_utf8_bytes_before_merging():
    vocab = {
        0: b"a",
        1: b"b",
    }
    tokenizer = tokenizer_module.Tokenizer(vocab=vocab, merges=[])

    assert tokenizer.encode("ab") == [0, 1]


def test_encode_applies_merges_in_rank_order():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"abc",
    }
    merges = [
        (b"a", b"b"),
        (b"ab", b"c"),
    ]
    tokenizer = tokenizer_module.Tokenizer(vocab=vocab, merges=merges)

    assert tokenizer.encode("abc") == [4]


def test_encode_keeps_special_token_as_one_token():
    special = "<|endoftext|>"
    vocab = {
        0: b"a",
        1: b"b",
        2: special.encode("utf-8"),
    }
    tokenizer = tokenizer_module.Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=[special],
    )

    ids = tokenizer.encode(f"a{special}b")

    assert ids == [0, 2, 1]
    assert tokenizer.decode(ids) == f"a{special}b"


def test_encode_prefers_longer_overlapping_special_token():
    short = "<|e|>"
    long = short + short
    vocab = {
        0: short.encode("utf-8"),
        1: long.encode("utf-8"),
    }
    tokenizer = tokenizer_module.Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=[short, long],
    )

    assert tokenizer.encode(long) == [1]


def test_encode_handles_repeated_missing_special_tokens():
    special = "<|sep|>"
    vocab = {0: b"a", 1: b"b"}
    tokenizer = tokenizer_module.Tokenizer(
        vocab=vocab,
        merges=[],
        special_tokens=[special],
    )

    ids = tokenizer.encode(f"{special}{special}")

    assert ids == [2, 2]
    assert tokenizer.decode(ids) == f"{special}{special}"
