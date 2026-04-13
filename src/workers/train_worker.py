"""QThread 기반 학습 워커."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.shap_utils import compute_and_save_shap
from src.core.training import generate_charts, train_models


class TrainWorker(QThread):
    """Tab2 – 모델 학습 워커."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)    # results dict
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params
            results = train_models(
                csv_path=p["csv_path"],
                target_col=p["target_col"],
                selected_models=p["selected_models"],
                task_type=p["task_type"],
                test_size=p["test_size"],
                cv_folds=p["cv_folds"],
                tune=p["tune"],
                n_iter=p["n_iter"],
                output_dir=p["output_dir"],
                progress_cb=self.progress.emit,
                log_cb=self.log.emit,
            )

            # Generate charts
            if results:
                chart_dir = os.path.join(p["output_dir"], "charts")
                self.log.emit("\n차트 생성 중...")
                generate_charts(results, p["task_type"], chart_dir)

            # SHAP analysis for training data
            if p.get("run_shap") and results:
                import pandas as pd
                df = pd.read_csv(p["csv_path"])
                exclude = {"fid", "FID", "grid_x", "grid_y", p["target_col"]}
                feat_cols = [c for c in df.columns if c not in exclude]
                X_all = df[feat_cols].dropna()

                shap_dir = os.path.join(p["output_dir"], "shap", "train")
                self.log.emit("\nSHAP 분석(학습) 시작...")
                for mname, res in results.items():
                    compute_and_save_shap(
                        model=res["model"],
                        X=X_all,
                        feature_cols=feat_cols,
                        model_name=mname,
                        output_dir=shap_dir,
                        mode="train",
                        log_cb=self.log.emit,
                    )

            self.finished.emit(results)

        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")
