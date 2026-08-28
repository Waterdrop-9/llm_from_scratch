import json

import numpy as np

from src.train_tinystories import main


def test_real_training_entrypoint_writes_metrics_and_checkpoint(tmp_path):
    train_path = tmp_path / "train.bin"
    valid_path = tmp_path / "valid.bin"
    output_root = tmp_path / "runs"
    np.tile(np.arange(8, dtype=np.uint16), 16).tofile(train_path)
    np.tile(np.arange(8, dtype=np.uint16), 16).tofile(valid_path)

    main(
        [
            "train",
            "--run-name",
            "test",
            "--train-data",
            str(train_path),
            "--valid-data",
            str(valid_path),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--vocab-size",
            "8",
            "--context-length",
            "4",
            "--d-model",
            "8",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--d-ff",
            "16",
            "--batch-size",
            "2",
            "--max-steps",
            "2",
            "--warmup-steps",
            "1",
            "--log-interval",
            "1",
            "--validation-interval",
            "1",
            "--validation-batches",
            "1",
        ]
    )

    run_directory = output_root / "test"
    records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]

    assert (run_directory / "checkpoint.pt").is_file()
    assert records[0]["event"] == "config"
    assert records[0]["parameter_count"] > 0
    assert any(record["event"] == "train" for record in records)
    assert any(record["event"] == "summary" for record in records)
    assert records[-1]["event"] == "checkpoint"
