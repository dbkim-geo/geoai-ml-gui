"""QThread 기반 전처리 워커."""
from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.preprocessing import preprocess_prediction, preprocess_training


class TrainPreprocessWorker(QThread):
    """Tab1 – 학습 데이터 전처리 워커."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)   # output CSV path
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params
            preprocess_training(
                point_path=p["point_path"],
                raster_configs=p["raster_configs"],
                buffer_sizes=p["buffer_sizes"],
                buffer_method=p["buffer_method"],
                target_col=p["target_col"],
                output_csv=p["output_csv"],
                progress_cb=self.progress.emit,
                log_cb=self.log.emit,
            )
            self.finished.emit(p["output_csv"])
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class PredPreprocessWorker(QThread):
    """Tab3 – 예측 데이터 전처리 워커."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str, dict)   # (csv_path, grid_meta)
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params
            _, grid_meta = preprocess_prediction(
                raster_configs=p["raster_configs"],
                buffer_sizes=p["buffer_sizes"],
                buffer_method=p["buffer_method"],
                extent=p["extent"],
                resolution=p.get("resolution"),
                output_csv=p["output_csv"],
                progress_cb=self.progress.emit,
                log_cb=self.log.emit,
            )
            self.finished.emit(p["output_csv"], grid_meta)
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
