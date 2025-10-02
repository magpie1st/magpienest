from __future__ import annotations

import threading
import logging
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
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "BookWorker init: chapters=%d out_dir=%s speaker=%s",
            len(getattr(plan, "chapters", [])),
            getattr(plan, "out_dir", "?"),
            getattr(plan, "speaker_wav", None),
        )

    def run(self) -> None:
        self.started.emit()
        self._logger.info(
            "BookWorker started: title=%s chapters=%d output=%s",
            getattr(self.plan.book, "title", "?"),
            len(getattr(self.plan, "chapters", [])),
            getattr(self.plan, "out_dir", "?"),
        )
        try:
            path = synthesize_book(
                self.plan,
                self._on_progress,
                self.cancel_event,
                log=self.log.emit,
            )
            self._logger.info("BookWorker finished successfully: %s", path)
            self.finished.emit(str(path))
        except Exception as exc:  # pragma: no cover
            self._logger.exception("Book synthesis failed")
            self.failed.emit(str(exc))

    def _on_progress(self, ci: int, ct: int, li: int, lt: int) -> None:
        self.progress.emit(ci, ct, li, lt)
