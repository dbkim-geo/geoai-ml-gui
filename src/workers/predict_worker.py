"""QThread 기반 예측 워커."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.prediction import load_grid_meta, predict_and_save
from src.core.shap_utils import compute_and_save_shap


class PredictWorker(QThread):
    """Tab4 – 예측 워커."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)    # output raster path
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params

            predict_and_save(
                csv_path=p["csv_path"],
                model_path=p["model_path"],
                output_raster=p["output_raster"],
                grid_meta=p.get("grid_meta"),
                log_cb=self.log.emit,
                progress_cb=self.progress.emit,
            )

            # SHAP for prediction data
            if p.get("run_shap"):
                import joblib
                import pandas as pd

                model = joblib.load(p["model_path"])
                df = pd.read_csv(p["csv_path"])
                exclude = {"grid_x", "grid_y"}
                feat_cols = [c for c in df.columns if c not in exclude]
                X = df[feat_cols].dropna()

                shap_dir = os.path.join(
                    os.path.dirname(p["output_raster"]), "shap", "predict"
                )
                self.log.emit("\nSHAP 분석(예측) 시작...")
                model_name = os.path.splitext(os.path.basename(p["model_path"]))[0]
                compute_and_save_shap(
                    model=model,
                    X=X,
                    feature_cols=feat_cols,
                    model_name=model_name,
                    output_dir=shap_dir,
                    mode="predict",
                    log_cb=self.log.emit,
                )

            self.finished.emit(p["output_raster"])

        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
