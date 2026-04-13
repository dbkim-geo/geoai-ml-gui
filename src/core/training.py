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
    feature_cols = [c for c in df.columns if c not in exclude]
    log(f"특성 변수: {len(feature_cols)}개  |  종속변수: '{target_col}'")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop NaN rows
    valid = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[valid], y[valid]
    log(f"유효 샘플: {len(X)}개 (NaN 제거 후)")

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
