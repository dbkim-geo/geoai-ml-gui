"""QThread 기반 학습 워커."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.shap_utils import compute_and_save_shap
from src.core.training import (
    generate_charts,
    parse_distances,
    select_mgwr_features,
    train_models,
    train_per_distance,
)


class TrainWorker(QThread):
    """Tab2 – 모델 학습 워커.

    distance_mode:
        'none'         – 전체 거리 통합 (기존 방식)
        'per_distance' – 거리별 독립 모델
        'mgwr'         – 변수별 최적 거리 자동 선택 후 단일 모델
    """

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)    # results dict  (per_distance → {'_meta': ..., dist: {...}})
    error = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            p = self.params
            mode = p.get("distance_mode", "none")

            if mode == "per_distance":
                self._run_per_distance(p)
            elif mode == "mgwr":
                self._run_mgwr(p)
            else:
                self._run_standard(p)

        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Standard (all distances combined)
    # ------------------------------------------------------------------
    def _run_standard(self, p: dict):
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

        if results:
            chart_dir = os.path.join(p["output_dir"], "charts")
            self.log.emit("\n차트 생성 중...")
            generate_charts(results, p["task_type"], chart_dir)

        if p.get("run_shap") and results:
            self._run_shap_train(p, results)

        self.finished.emit(results)

    # ------------------------------------------------------------------
    # Per-distance mode
    # ------------------------------------------------------------------
    def _run_per_distance(self, p: dict):
        self.log.emit("[거리별 모드] 거리별 독립 모델 학습")
        all_results = train_per_distance(
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

        if p.get("run_shap"):
            import pandas as pd
            df = pd.read_csv(p["csv_path"])
            exclude = {"fid", "FID", "grid_x", "grid_y", p["target_col"]}
            all_features = [c for c in df.columns if c not in exclude]
            dist_map = parse_distances(all_features)

            for dist, results in all_results.items():
                if not results:
                    continue
                cols = dist_map.get(dist, [])
                X = df[cols].fillna(df[cols].median())
                shap_dir = os.path.join(p["output_dir"], f"{dist}m", "shap", "train")
                self.log.emit(f"\nSHAP 분석 ({dist}m)...")
                for mname, res in results.items():
                    compute_and_save_shap(
                        model=res["model"], X=X, feature_cols=cols,
                        model_name=mname, output_dir=shap_dir,
                        mode="train", log_cb=self.log.emit,
                    )

        # Wrap with meta for UI to distinguish
        wrapped = {"_mode": "per_distance", "_data": all_results}
        self.finished.emit(wrapped)

    # ------------------------------------------------------------------
    # MGWR mode
    # ------------------------------------------------------------------
    def _run_mgwr(self, p: dict):
        import pandas as pd

        self.log.emit("[MGWR 모드] 변수별 최적 거리 선택 중...")

        df = pd.read_csv(p["csv_path"])
        exclude = {"fid", "FID", "grid_x", "grid_y", p["target_col"]}
        all_features = [c for c in df.columns if c not in exclude]
        y = df[p["target_col"]].dropna()
        X_all = df[all_features].loc[y.index].fillna(df[all_features].median())

        best_cols, selection_df = select_mgwr_features(
            X=X_all, y=y,
            feature_cols=all_features,
            task_type=p["task_type"],
            log_cb=self.log.emit,
        )

        # Save selection table
        os.makedirs(p["output_dir"], exist_ok=True)
        sel_path = os.path.join(p["output_dir"], "scale_opt_selection.csv")
        selection_df.to_csv(sel_path, index=False, encoding="utf-8-sig")
        self.log.emit(f"\n최적 거리 선택 결과: {sel_path}")
        self.log.emit(f"선택된 변수 수: {len(best_cols)}")

        mgwr_dir = os.path.join(p["output_dir"], "scale_opt")
        results = train_models(
            csv_path=p["csv_path"],
            target_col=p["target_col"],
            selected_models=p["selected_models"],
            task_type=p["task_type"],
            test_size=p["test_size"],
            cv_folds=p["cv_folds"],
            tune=p["tune"],
            n_iter=p["n_iter"],
            output_dir=mgwr_dir,
            feature_cols_override=best_cols,
            progress_cb=self.progress.emit,
            log_cb=self.log.emit,
        )

        if results:
            chart_dir = os.path.join(mgwr_dir, "charts")
            generate_charts(results, p["task_type"], chart_dir)

        if p.get("run_shap") and results:
            self._run_shap_train(
                {**p, "output_dir": mgwr_dir},
                results,
                feature_cols_override=best_cols,
            )

        wrapped = {"_mode": "mgwr", "_data": results, "_selection": selection_df}
        self.finished.emit(wrapped)

    # ------------------------------------------------------------------
    # SHAP helper
    # ------------------------------------------------------------------
    def _run_shap_train(self, p: dict, results: dict, feature_cols_override=None):
        import pandas as pd
        df = pd.read_csv(p["csv_path"])
        exclude = {"fid", "FID", "grid_x", "grid_y", p["target_col"]}
        feat_cols = feature_cols_override or [c for c in df.columns if c not in exclude]
        X_all = df[feat_cols].fillna(df[feat_cols].median())

        shap_dir = os.path.join(p["output_dir"], "shap", "train")
        self.log.emit("\nSHAP 분석(학습) 시작...")
        for mname, res in results.items():
            compute_and_save_shap(
                model=res["model"], X=X_all, feature_cols=feat_cols,
                model_name=mname, output_dir=shap_dir,
                mode="train", log_cb=self.log.emit,
            )
