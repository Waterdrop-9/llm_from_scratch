from __future__ import annotations

from collections import Counter
from pathlib import Path

import regex


# GPT-2/CS336 pre-tokenization pattern. BPE merges are learned inside each
# match, so unrelated words, punctuation, and whitespace cannot merge across
# their boundaries.
_PRETOKENIZATION_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge_pair_in_sequence(
    sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of pair from left to right."""
    merged_token = pair[0] + pair[1]
    merged_sequence: list[bytes] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            merged_sequence.append(merged_token)
            index += 2
        else:
            merged_sequence.append(sequence[index])
            index += 1
    return tuple(merged_sequence)


def _count_pairs(
    sequence_frequencies: dict[tuple[bytes, ...], int],
) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent pairs, weighted by each sequence's corpus frequency."""
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for sequence, frequency in sequence_frequencies.items():
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] += frequency
    return pair_counts


def _add_special_tokens(
    vocab: dict[int, bytes],
    special_tokens: list[str],
) -> None:
    """Append special tokens that are not already present in the vocabulary."""
    existing_tokens = set(vocab.values())
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in existing_tokens:
            vocab[len(vocab)] = token_bytes
            existing_tokens.add(token_bytes)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a CS336-style byte-level BPE tokenizer on a text corpus."""
    # The caller may repeat a special token; it should occupy only one vocab
    # entry and one alternative in the segmentation regex.
    unique_special_tokens = list(dict.fromkeys(special_tokens))

    # IDs 0..255 are reserved for every possible byte. special tokens are
    # part of vocab_size, so merges can only consume the remaining slots.
    base_byte_tokens = {bytes([token_id]) for token_id in range(256)}
    special_token_bytes = {
        token.encode("utf-8")
        for token in unique_special_tokens
    }
    minimum_vocab_size = 256 + len(special_token_bytes - base_byte_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {minimum_vocab_size}, got {vocab_size}."
        )

    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    _add_special_tokens(vocab, unique_special_tokens)

    try:
        text = Path(input_path).read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Training corpus not found: {input_path}") from error

    # Split on special tokens before pre-tokenization. The separators are not
    # included in training sequences because they are already atomic tokens;
    # this prevents a merge from crossing a special-token boundary.
    if unique_special_tokens:
        special_pattern = regex.compile(
            "|".join(
                regex.escape(token)
                for token in sorted(unique_special_tokens, key=len, reverse=True)
            )
        )
        ordinary_chunks = special_pattern.split(text)
    else:
        ordinary_chunks = [text]

    # Store each distinct pre-tokenized word once and retain its frequency.
    # A sequence is initially represented as one byte per token.
    sequence_frequencies: Counter[tuple[bytes, ...]] = Counter()
    for chunk in ordinary_chunks:
        for piece in _PRETOKENIZATION_PATTERN.findall(chunk):
            sequence = tuple(bytes([byte]) for byte in piece.encode("utf-8"))
            if sequence:
                sequence_frequencies[sequence] += 1

    pair_counts = _count_pairs(sequence_frequencies)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        # max() first compares frequency and then compares byte tuples. The
        # latter implements CS336's lexicographically-greater tie-break rule.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected = [
            (sequence, frequency)
            for sequence, frequency in sequence_frequencies.items()
            if best_pair in zip(sequence, sequence[1:])
        ]

        # Remove all old contributions first. This handles collisions where
        # two different sequences become the same sequence after a merge.
        for sequence, frequency in affected:
            del sequence_frequencies[sequence]
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] -= frequency
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

        for sequence, frequency in affected:
            merged_sequence = _merge_pair_in_sequence(sequence, best_pair)
            sequence_frequencies[merged_sequence] += frequency
            for pair in zip(merged_sequence, merged_sequence[1:]):
                pair_counts[pair] += frequency

    return vocab, merges
