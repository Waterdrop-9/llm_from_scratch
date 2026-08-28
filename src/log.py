import json
import time
from pathlib import Path
from types import TracebackType
from typing import Self


class Logger:
    """Write experiment events to both stdout and a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._start_time = time.perf_counter()

    def log(self, event: str, **fields: object) -> dict[str, object]:
        if self._stream.closed:
            raise ValueError("Cannot write to a closed experiment logger.")

        record = {
            **fields,
            "event": event,
            "elapsed_seconds": time.perf_counter() - self._start_time,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(line)
        self._stream.write(line + "\n")
        self._stream.flush()
        return record

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


# Backward-compatible name used by the existing tests and earlier notes.
ExperimentLogger = Logger
