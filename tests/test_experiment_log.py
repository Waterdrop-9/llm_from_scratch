import json
from pathlib import Path

import pytest

from src.log import ExperimentLogger


def test_experiment_logger_writes_jsonl_and_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    log_path = tmp_path / "run" / "metrics.jsonl"

    with ExperimentLogger(log_path) as logger:
        config_record = logger.log("config", seed=42, device="cpu")
        train_record = logger.log(
            "train",
            step=10,
            tokens_processed=5120,
            train_loss=2.5,
            learning_rate=1e-3,
        )

    stored_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    stdout_records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]

    assert stored_records == stdout_records
    assert stored_records[0]["event"] == "config"
    assert stored_records[1]["event"] == "train"
    assert stored_records[1]["step"] == 10
    assert stored_records[1]["tokens_processed"] == 5120
    assert config_record["elapsed_seconds"] >= 0
    assert train_record["elapsed_seconds"] >= config_record["elapsed_seconds"]


def test_experiment_logger_rejects_writes_after_close(tmp_path: Path):
    logger = ExperimentLogger(tmp_path / "metrics.jsonl")
    logger.close()

    with pytest.raises(ValueError, match="closed"):
        logger.log("train", step=1)
