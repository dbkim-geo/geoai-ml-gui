"""
SHAP 분석 모듈.

학습(train) 및 예측(predict) 데이터 각각에 대해 SHAP 값을 계산하고
figure를 저장한다.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def _get_explainer(model, X_bg: pd.DataFrame):
    """모델 종류에 따라 적절한 explainer 선택."""
    model_type = type(model).__name__.lower()
    tree_types = ("forest", "tree", "boost", "xgb", "lgb", "cat", "gradient", "ada")
    linear_types = ("ridge", "lasso", "elastic", "logistic", "linear")

    if any(t in model_type for t in tree_types):
        return shap.TreeExplainer(model)
    elif any(t in model_type for t in linear_types):
        return shap.LinearExplainer(model, X_bg)
    else:
        bg = shap.sample(X_bg, min(50, len(X_bg)))
        return shap.KernelExplainer(model.predict, bg)


def compute_and_save_shap(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    output_dir: str,
    mode: str = "train",   # 'train' | 'predict'
    max_samples: int = 1000,
    log_cb: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    SHAP 값을 계산하고 summary / importance plot을 저장.

    Parameters
    ----------
    mode : 'train' 또는 'predict' — 저장 경로와 figure 제목에 반영됨.

    Returns
    -------
    np.ndarray or None
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    if not _HAS_SHAP:
        log("[SHAP 오류] shap 패키지가 설치되지 않았습니다. pip install shap")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Sample if too large
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42).reset_index(drop=True)
        log(f"  {max_samples}개 샘플링 (전체 {len(X):,}개 중)")
    else:
        X_sample = X.reset_index(drop=True)

    safe_name = model_name.replace(" ", "_")
    tag = "train" if mode == "train" else "predict"

    log(f"  SHAP 계산: {model_name} [{mode}] ...")
    try:
        explainer = _get_explainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Multi-class → average over classes
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)

        # --- Summary beeswarm plot ---
        plt.figure(figsize=(12, max(6, len(feature_cols) * 0.3 + 2)))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.title(f"{model_name} – SHAP Summary [{tag.upper()}]", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p1 = os.path.join(output_dir, f"{safe_name}_{tag}_shap_summary.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  저장: {p1}")

        # --- Bar (mean |SHAP|) ---
        plt.figure(figsize=(10, max(5, len(feature_cols) * 0.25 + 2)))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                          plot_type="bar", show=False)
        plt.title(f"{model_name} – SHAP Importance [{tag.upper()}]", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p2 = os.path.join(output_dir, f"{safe_name}_{tag}_shap_importance.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  저장: {p2}")

        return shap_values

    except Exception as exc:
        import traceback
        log(f"  [SHAP 오류] {exc}")
        log(traceback.format_exc())
        return None
