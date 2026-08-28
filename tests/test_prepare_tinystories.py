import numpy as np

from src.train_tinystories import main


def test_cli_trains_then_reuses_the_same_tokenizer(tmp_path):
    tokenizer_corpus = tmp_path / "tokenizer.txt"
    train_text = tmp_path / "train.txt"
    valid_text = tmp_path / "valid.txt"
    tokenizer_path = tmp_path / "tokenizer.pkl"
    train_tokens = tmp_path / "train.bin"
    valid_tokens = tmp_path / "valid.bin"

    tokenizer_corpus.write_text("abab abab", encoding="utf-8")
    train_text.write_text("abab", encoding="utf-8")
    valid_text.write_text("baba", encoding="utf-8")

    main(
        [
            "prepare",
            "--tokenizer-corpus",
            str(tokenizer_corpus),
            "--tokenizer-path",
            str(tokenizer_path),
            "--input",
            str(train_text),
            "--output",
            str(train_tokens),
            "--vocab-size",
            "258",
            "--buffer-size",
            "2",
        ]
    )
    main(
        [
            "prepare",
            "--tokenizer-path",
            str(tokenizer_path),
            "--input",
            str(valid_text),
            "--output",
            str(valid_tokens),
        ]
    )

    assert tokenizer_path.is_file()
    assert np.fromfile(train_tokens, dtype=np.uint16).size > 0
    assert np.fromfile(valid_tokens, dtype=np.uint16).size > 0
