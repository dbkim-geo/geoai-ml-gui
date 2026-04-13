"""
ML 학습 핵심 모듈.

10+ 모델 정의, RandomizedSearchCV 튜닝, CV 성능 평가, 모델/결과 저장.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


def _setup_korean_font():
    """OS별 한글 폰트를 찾아 matplotlib에 설정."""
    import platform
    candidates = {
        "Windows": ["Malgun Gothic", "맑은 고딕"],
        "Darwin":  ["AppleGothic", "NanumGothic"],
        "Linux":   ["NanumGothic", "NanumBarunGothic", "UnDotum"],
    }.get(platform.system(), ["NanumGothic"])

    available = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False   # 최소한 마이너스 부호만 고정


_setup_korean_font()

# Optional libraries
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_REG_MODELS: dict[str, dict] = {
    "Random Forest": {
        "cls": RandomForestRegressor,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1},
    },
    "Extra Trees": {
        "cls": ExtraTreesRegressor,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1},
    },
    "Gradient Boosting": {
        "cls": GradientBoostingRegressor,
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
        "kwargs": {"random_state": 42},
    },
    "AdaBoost": {
        "cls": AdaBoostRegressor,
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
        "kwargs": {"random_state": 42},
    },
    "Decision Tree": {
        "cls": DecisionTreeRegressor,
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "kwargs": {"random_state": 42},
    },
    "SVR": {
        "cls": SVR,
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "kwargs": {},
    },
    "KNN": {
        "cls": KNeighborsRegressor,
        "params": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
        },
        "kwargs": {"n_jobs": -1},
    },
    "Ridge": {
        "cls": Ridge,
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "kwargs": {},
    },
    "Lasso": {
        "cls": Lasso,
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "kwargs": {"random_state": 42},
    },
    "ElasticNet": {
        "cls": ElasticNet,
        "params": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "kwargs": {"random_state": 42},
    },
    "MLP": {
        "cls": MLPRegressor,
        "params": {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
        },
        "kwargs": {"max_iter": 500, "random_state": 42},
    },
}

_CLF_MODELS: dict[str, dict] = {
    "Random Forest": {
        "cls": RandomForestClassifier,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1},
    },
    "Extra Trees": {
        "cls": ExtraTreesClassifier,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1},
    },
    "Gradient Boosting": {
        "cls": GradientBoostingClassifier,
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "kwargs": {"random_state": 42},
    },
    "AdaBoost": {
        "cls": AdaBoostClassifier,
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
        },
        "kwargs": {"random_state": 42},
    },
    "Decision Tree": {
        "cls": DecisionTreeClassifier,
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "kwargs": {"random_state": 42},
    },
    "SVC": {
        "cls": SVC,
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
        },
        "kwargs": {"probability": True, "random_state": 42},
    },
    "KNN": {
        "cls": KNeighborsClassifier,
        "params": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
        },
        "kwargs": {"n_jobs": -1},
    },
    "Logistic Regression": {
        "cls": LogisticRegression,
        "params": {"C": [0.01, 0.1, 1.0, 10.0]},
        "kwargs": {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
    },
    "MLP": {
        "cls": MLPClassifier,
        "params": {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
        },
        "kwargs": {"max_iter": 500, "random_state": 42},
    },
}

# Inject optional models
if _HAS_XGB:
    _REG_MODELS["XGBoost"] = {
        "cls": xgb.XGBRegressor,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1, "verbosity": 0},
    }
    _CLF_MODELS["XGBoost"] = {
        "cls": xgb.XGBClassifier,
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1, "verbosity": 0, "use_label_encoder": False, "eval_metric": "logloss"},
    }

if _HAS_LGB:
    _REG_MODELS["LightGBM"] = {
        "cls": lgb.LGBMRegressor,
        "params": {
            "n_estimators": [100, 200, 300],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1, "verbose": -1},
    }
    _CLF_MODELS["LightGBM"] = {
        "cls": lgb.LGBMClassifier,
        "params": {
            "n_estimators": [100, 200, 300],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "kwargs": {"random_state": 42, "n_jobs": -1, "verbose": -1},
    }

if _HAS_CAT:
    _REG_MODELS["CatBoost"] = {
        "cls": CatBoostRegressor,
        "params": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "kwargs": {"random_seed": 42, "verbose": 0},
    }
    _CLF_MODELS["CatBoost"] = {
        "cls": CatBoostClassifier,
        "params": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "kwargs": {"random_seed": 42, "verbose": 0},
    }


def get_available_models(task_type: str) -> list[str]:
    registry = _REG_MODELS if task_type == "regression" else _CLF_MODELS
    return list(registry.keys())


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_models(
    csv_path: str,
    target_col: str,
    selected_models: list[str],
    task_type: str = "regression",
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune: bool = True,
    n_iter: int = 10,
    output_dir: Optional[str] = None,
    feature_cols_override: Optional[list[str]] = None,
    progress_cb: Optional[Callable] = None,
    log_cb: Optional[Callable] = None,
) -> dict:
    """
    선택한 모델들을 학습하고 결과를 반환.

    Returns
    -------
    dict: {model_name: {model, model_path, metrics, feature_importance, y_test, y_pred, feature_cols}}
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def prog(v: float):
        if progress_cb:
            progress_cb(int(v))

    log(f"데이터 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    exclude = {"fid", "FID", "grid_x", "grid_y", target_col}
    if feature_cols_override is not None:
        feature_cols = [c for c in feature_cols_override if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in exclude]
    log(f"특성 변수: {len(feature_cols)}개  |  종속변수: '{target_col}'")

    if target_col not in df.columns:
        raise ValueError(f"종속변수 컬럼 '{target_col}'을 CSV에서 찾을 수 없습니다.\n"
                         f"사용 가능한 컬럼: {list(df.columns)}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ① 종속변수 NaN 행만 제거 (필수)
    y_valid = ~y.isna()
    X, y = X[y_valid], y[y_valid]

    # ② 독립변수 NaN → 컬럼 중앙값으로 대체 (데이터 최대 보존)
    col_medians = X.median(numeric_only=True)
    X = X.fillna(col_medians)

    # ③ 중앙값도 NaN인 컬럼(전체 결측) 제거
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        log(f"[경고] 전체 결측 컬럼 제거: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
        feature_cols = list(X.columns)

    if len(X) < 5:
        raise ValueError(
            f"유효 샘플이 너무 적습니다 ({len(X)}개).\n"
            "포인트 데이터와 래스터의 공간 범위가 겹치는지, "
            "종속변수 컬럼이 올바른지 확인하세요."
        )
    log(f"유효 샘플: {len(X)}개")

    # Encode labels for classification
    le: Optional[LabelEncoder] = None
    if task_type == "classification" and (y.dtype == object or str(y.dtype) == "category"):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    log(f"학습: {len(X_train)}개  |  테스트: {len(X_test)}개")

    registry = _REG_MODELS if task_type == "regression" else _CLF_MODELS
    results: dict = {}
    total = len(selected_models)

    for i, model_name in enumerate(selected_models):
        if model_name not in registry:
            log(f"[건너뜀] 알 수 없는 모델: {model_name}")
            continue

        log(f"\n[{i + 1}/{total}] {model_name} 학습 중...")
        cfg = registry[model_name]
        ModelCls = cfg["cls"]
        param_grid = cfg["params"]
        kwargs = cfg["kwargs"]

        try:
            base_model = ModelCls(**kwargs)

            if tune and param_grid:
                scoring = (
                    "neg_root_mean_squared_error"
                    if task_type == "regression"
                    else "f1_weighted"
                )
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=min(n_iter, 20),
                    cv=min(cv_folds, 3),
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42,
                    refit=True,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                log(f"  최적 파라미터: {search.best_params_}")
            else:
                model = base_model
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # --- Metrics ---
            if task_type == "regression":
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                metrics = {
                    "RMSE": rmse,
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAPE": float(
                        np.mean(np.abs((y_test.values - y_pred) / (np.abs(y_test.values) + 1e-8))) * 100
                    ),
                }
            else:
                metrics = {
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "F1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "Precision_w": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "Recall_w": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                }
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                        if y_prob.shape[1] == 2:
                            metrics["ROC_AUC"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                        else:
                            metrics["ROC_AUC"] = float(
                                roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                            )
                except Exception:
                    pass

            # Cross-validation
            cv_sc = "neg_root_mean_squared_error" if task_type == "regression" else "f1_weighted"
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=cv_sc, n_jobs=-1)
            metrics["CV_mean"] = float(cv_scores.mean())
            metrics["CV_std"] = float(cv_scores.std())

            # Feature importance
            fi: Optional[pd.Series] = None
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                fi = pd.Series(coef, index=feature_cols).sort_values(ascending=False)

            # Save model
            model_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe = model_name.replace(" ", "_")
                model_path = os.path.join(output_dir, f"{safe}.pkl")
                joblib.dump(model, model_path)
                log(f"  모델 저장: {model_path}")

            log(f"  결과: { {k: f'{v:.4f}' for k, v in metrics.items()} }")

            results[model_name] = {
                "model": model,
                "model_path": model_path,
                "metrics": metrics,
                "feature_importance": fi,
                "feature_cols": feature_cols,
                "y_test": y_test.values,
                "y_pred": y_pred,
                "label_encoder": le,
                "task_type": task_type,
            }

        except Exception as exc:
            import traceback
            log(f"  [오류] {model_name}: {exc}")
            log(traceback.format_exc())

        prog((i + 1) / total * 100)

    # Save metrics summary
    if output_dir and results:
        rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "metrics_summary.csv"),
            index=False, encoding="utf-8-sig"
        )
        log(f"\n성능 요약 저장: {os.path.join(output_dir, 'metrics_summary.csv')}")

    log("\n학습 완료!")
    return results


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_charts(results: dict, task_type: str, output_dir: str) -> list[str]:
    """성능 비교 차트 생성 후 경로 목록 반환."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    names = list(results.keys())

    # --- 1. Performance comparison bar chart ---
    metrics_keys = (
        ["RMSE", "MAE", "R2"] if task_type == "regression"
        else ["Accuracy", "F1_weighted", "Precision_w", "Recall_w"]
    )
    n = len(metrics_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, mkey in zip(axes, metrics_keys):
        vals = [results[m]["metrics"].get(mkey, np.nan) for m in names]
        bars = ax.bar(names, vals, color="steelblue", edgecolor="black", width=0.6)
        ax.set_title(mkey, fontsize=13, fontweight="bold")
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(mkey)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7,
                )
    plt.suptitle("모델 성능 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # --- 2. Feature importance (top 20, per model) ---
    for mname, res in results.items():
        fi = res.get("feature_importance")
        if fi is not None:
            top = fi.head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            top.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"{mname} – Feature Importance (Top 20)", fontsize=13, fontweight="bold")
            ax.invert_yaxis()
            plt.tight_layout()
            safe = mname.replace(" ", "_")
            p = os.path.join(output_dir, f"{safe}_feature_importance.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            saved.append(p)

    # --- 3. Actual vs Predicted scatter (regression) ---
    if task_type == "regression":
        nc = min(3, len(results))
        nr = (len(results) + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
        axes_flat = np.array(axes).flatten()
        for idx, (mname, res) in enumerate(results.items()):
            ax = axes_flat[idx]
            yt, yp = res["y_test"], res["y_pred"]
            ax.scatter(yt, yp, alpha=0.4, s=8, color="steelblue")
            lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lim, lim, "r--", lw=1.5)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            r2 = res["metrics"].get("R2", float("nan"))
            ax.set_title(f"{mname}\nR²={r2:.4f}", fontsize=11)
        for j in range(len(results), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.tight_layout()
        p = os.path.join(output_dir, "actual_vs_predicted.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Distance-based modeling utilities
# ---------------------------------------------------------------------------

import re as _re


def parse_distances(feature_cols: list[str]) -> dict[int, list[str]]:
    """
    컬럼명 패턴 `{name}_{distance}m_...` 에서 거리를 파싱.

    Returns
    -------
    dict[int, list[str]] : {거리(m): [해당 거리 컬럼 목록]}  (오름차순 정렬)
    """
    result: dict[int, list[str]] = {}
    for col in feature_cols:
        m = _re.search(r"_(\d+)m_", col)
        if m:
            d = int(m.group(1))
            result.setdefault(d, []).append(col)
    return dict(sorted(result.items()))


def select_scale_opt_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    task_type: str = "regression",
    method: str = "corr",          # "corr" | "model"
    log_cb: Optional[Callable] = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Scale of Effect: 각 래스터 변수에 대해 타겟과 관련성이 가장 높은 버퍼 거리 선택.

    Parameters
    ----------
    method : str
        "corr"  – 단변량 상관계수 (회귀: Spearman, 분류: Mutual Information).
                  빠르지만 변수 간 상호작용을 고려하지 않음.
        "model" – Random Forest Feature Importance 기반.
                  변수 간 상호작용을 반영하며 더 정확하나 시간이 걸림.

    Returns
    -------
    (best_cols, selection_df)
        best_cols      – 선택된 컬럼 목록
        selection_df   – {variable, distance_m, score, selected, method} DataFrame
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    # Group columns by variable name → {var_name: {distance: [cols]}}
    var_groups: dict[str, dict[int, list[str]]] = {}
    for col in feature_cols:
        m = _re.match(r"^(.+?)_(\d+)m_", col)
        if m:
            vname, dist = m.group(1), int(m.group(2))
            var_groups.setdefault(vname, {}).setdefault(dist, []).append(col)

    # ── Score computation ──────────────────────────────────────────────
    if method == "model":
        log("  [Scale Opt] 방법: Random Forest Feature Importance")
        score_map = _score_by_model(X, y, feature_cols, task_type, log)
    else:
        log("  [Scale Opt] 방법: 상관계수 (Spearman / Mutual Information)")
        score_map = _score_by_corr(X, y, feature_cols, task_type)

    # ── Select best distance per variable ─────────────────────────────
    best_cols: list[str] = []
    records: list[dict] = []

    for vname, dist_map in var_groups.items():
        best_dist, best_score = None, -np.inf

        for dist, cols in sorted(dist_map.items()):
            score = float(np.mean([score_map.get(c, 0.0) for c in cols]))
            records.append({
                "variable": vname, "distance_m": dist,
                "score": score, "selected": False, "method": method,
            })
            if score > best_score:
                best_score, best_dist = score, dist

        if best_dist is not None:
            best_cols.extend(dist_map[best_dist])
            log(f"  {vname}: 최적 거리 = {best_dist}m  (score={best_score:.4f})")
            for rec in records:
                if rec["variable"] == vname and rec["distance_m"] == best_dist:
                    rec["selected"] = True

    selection_df = pd.DataFrame(records)
    return best_cols, selection_df


def _score_by_corr(
    X: pd.DataFrame, y: pd.Series,
    feature_cols: list[str], task_type: str,
) -> dict[str, float]:
    """단변량 상관계수로 각 컬럼의 score 계산."""
    from scipy.stats import spearmanr

    scores = {}
    for col in feature_cols:
        if col not in X.columns:
            scores[col] = 0.0
            continue
        xc = X[col].fillna(0)
        try:
            if task_type == "regression":
                r, _ = spearmanr(xc, y, nan_policy="omit")
                scores[col] = abs(float(r)) if not np.isnan(r) else 0.0
            else:
                from sklearn.feature_selection import mutual_info_classif
                mi = mutual_info_classif(
                    xc.values.reshape(-1, 1), y.values, random_state=42
                )[0]
                scores[col] = float(mi)
        except Exception:
            scores[col] = 0.0
    return scores


def _score_by_model(
    X: pd.DataFrame, y: pd.Series,
    feature_cols: list[str], task_type: str,
    log_fn,
) -> dict[str, float]:
    """Random Forest Feature Importance로 각 컬럼의 score 계산."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X_filled = X[feature_cols].fillna(X[feature_cols].median())
    log_fn("  RF 학습 중 (n_estimators=100)...")

    if task_type == "regression":
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X_filled, y)
    fi = rf.feature_importances_
    return dict(zip(feature_cols, fi.tolist()))


def train_per_distance(
    csv_path: str,
    target_col: str,
    selected_models: list[str],
    task_type: str = "regression",
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune: bool = True,
    n_iter: int = 10,
    output_dir: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
    log_cb: Optional[Callable] = None,
) -> dict[int, dict]:
    """
    버퍼 거리별 독립 모델 학습.

    Returns
    -------
    dict[int, dict] : {거리(m): train_models() 결과}
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def prog(v: float):
        if progress_cb:
            progress_cb(int(v))

    df = pd.read_csv(csv_path)
    exclude = {"fid", "FID", "grid_x", "grid_y", target_col}
    all_features = [c for c in df.columns if c not in exclude]

    dist_map = parse_distances(all_features)
    if not dist_map:
        raise ValueError(
            "컬럼명에서 버퍼 거리를 파싱할 수 없습니다.\n"
            "전처리 CSV가 올바른지 확인하세요 (예: slope_100m_mean)."
        )

    log(f"감지된 거리: {list(dist_map.keys())} m")
    all_results: dict[int, dict] = {}
    n = len(dist_map)

    for i, (dist, cols) in enumerate(dist_map.items()):
        log(f"\n{'='*40}")
        log(f"  거리 {dist}m  ({len(cols)}개 변수)")
        log(f"{'='*40}")
        dist_dir = os.path.join(output_dir, f"{dist}m") if output_dir else None

        results = train_models(
            csv_path=csv_path,
            target_col=target_col,
            selected_models=selected_models,
            task_type=task_type,
            test_size=test_size,
            cv_folds=cv_folds,
            tune=tune,
            n_iter=n_iter,
            output_dir=dist_dir,
            feature_cols_override=cols,
            progress_cb=lambda v, _i=i: prog(_i / n * 100 + v / n),
            log_cb=log_cb,
        )

        if results and dist_dir:
            chart_dir = os.path.join(dist_dir, "charts")
            generate_charts(results, task_type, chart_dir)

        all_results[dist] = results

    # Cross-distance comparison chart
    if output_dir and all_results:
        _generate_distance_comparison(all_results, task_type, output_dir)

    prog(100)
    return all_results


def _generate_distance_comparison(
    all_results: dict[int, dict], task_type: str, output_dir: str
):
    """거리별 최고 성능 모델의 지표를 한 눈에 비교하는 차트."""
    os.makedirs(output_dir, exist_ok=True)

    primary = "R2" if task_type == "regression" else "F1_weighted"
    secondary = "RMSE" if task_type == "regression" else "Accuracy"

    distances, primary_vals, secondary_vals, best_models = [], [], [], []

    for dist, results in sorted(all_results.items()):
        if not results:
            continue
        # Best model by primary metric
        best = max(
            results.items(),
            key=lambda kv: kv[1]["metrics"].get(primary, -np.inf),
        )
        bname, bres = best
        distances.append(dist)
        primary_vals.append(bres["metrics"].get(primary, np.nan))
        secondary_vals.append(bres["metrics"].get(secondary, np.nan))
        best_models.append(bname)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    x = np.arange(len(distances))
    bars1 = ax1.bar(x, primary_vals, color="steelblue", edgecolor="black")
    ax1.set_ylabel(primary, fontsize=12)
    ax1.set_title("거리별 최고 모델 성능 비교", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}m\n({m})" for d, m in zip(distances, best_models)], fontsize=9)
    for bar, v in zip(bars1, primary_vals):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, secondary_vals, color="coral", edgecolor="black")
    ax2.set_ylabel(secondary, fontsize=12)
    ax2.set_xlabel("버퍼 거리 (m)", fontsize=12)
    for bar, v in zip(bars2, secondary_vals):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, "distance_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
