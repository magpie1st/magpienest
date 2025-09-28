from __future__ import annotations

import threading
from PySide6.QtCore import QObject, Signal

from audiobook.core.pipeline import synthesize_book


class BookWorker(QObject):
    started = Signal()
    progress = Signal(int, int, int, int)
    finished = Signal(str)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, plan, cancel_event: threading.Event) -> None:
        super().__init__()
        self.plan = plan
        self.cancel_event = cancel_event

    def run(self) -> None:
        self.started.emit()
        try:
            path = synthesize_book(
                self.plan,
                self._on_progress,
                self.cancel_event,
                log=self.log.emit,
            )
            self.finished.emit(str(path))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))

    def _on_progress(self, ci: int, ct: int, li: int, lt: int) -> None:
        self.progress.emit(ci, ct, li, lt)
