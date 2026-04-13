"""
예측 및 GeoTIFF 래스터 저장 모듈.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


def load_grid_meta(csv_path: str) -> Optional[dict]:
    """예측 CSV와 쌍으로 저장된 grid_meta JSON 로드."""
    meta_path = csv_path.replace(".csv", "_grid_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_and_save(
    csv_path: str,
    model_path: str,
    output_raster: str,
    grid_meta: Optional[dict] = None,
    log_cb: Optional[Callable] = None,
    progress_cb: Optional[Callable] = None,
) -> str:
    """
    학습된 모델로 예측 후 GeoTIFF 저장.

    Returns
    -------
    str : 저장된 래스터 경로
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def prog(v: float):
        if progress_cb:
            progress_cb(int(v))

    log(f"모델 로드: {model_path}")
    model = joblib.load(model_path)
    prog(10)

    log(f"예측 CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    # Separate coordinates from features
    coord_cols = {"grid_x", "grid_y"}
    feature_cols = [c for c in df.columns if c not in coord_cols]
    log(f"특성 변수: {len(feature_cols)}개  |  포인트: {len(df):,}개")

    X = df[feature_cols].copy()

    # 모델이 학습한 피처 목록에 맞춰 컬럼 정렬
    # (범주형 클래스가 학습/예측 영역마다 달라질 수 있음)
    if hasattr(model, "feature_names_in_"):
        train_cols = list(model.feature_names_in_)
    elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
        train_cols = model.get_booster().feature_names or []
    else:
        train_cols = []

    if train_cols:
        missing = [c for c in train_cols if c not in X.columns]
        extra   = [c for c in X.columns  if c not in train_cols]
        if missing:
            log(f"  [컬럼 정렬] 학습에 있었으나 예측 CSV에 없는 컬럼 {len(missing)}개 → 0 채움")
            for c in missing:
                X[c] = 0
        if extra:
            log(f"  [컬럼 정렬] 예측 CSV에만 있는 컬럼 {len(extra)}개 → 제거")
            X = X.drop(columns=extra)
        X = X[train_cols]  # 순서까지 일치시킴
        log(f"  피처 정렬 완료: {len(train_cols)}개")

    prog(20)

    # Handle NaN
    nan_mask = X.isna().any(axis=1)
    if nan_mask.sum() > 0:
        log(f"NaN 포함 행 {nan_mask.sum():,}개 → NoData 처리")

    predictions = np.full(len(df), np.nan, dtype=np.float32)
    valid = ~nan_mask
    if valid.any():
        log("예측 중...")
        predictions[valid] = model.predict(X[valid]).astype(np.float32)
    prog(70)

    # Grid meta
    if grid_meta is None:
        grid_meta = load_grid_meta(csv_path)
    if grid_meta is None:
        raise ValueError(
            "격자 메타데이터를 찾을 수 없습니다. "
            "예측 전처리 결과의 _grid_meta.json 파일이 필요합니다."
        )

    nx = grid_meta["nx"]
    ny = grid_meta["ny"]
    minx = grid_meta["minx"]
    maxy = grid_meta["maxy"]
    res = grid_meta["resolution"]
    crs = CRS.from_string(grid_meta["crs"])

    log(f"격자 크기: {nx} × {ny}")

    # Reshape — ys were created bottom→top, raster is top→bottom
    pred_grid = predictions.reshape(ny, nx)[::-1, :]

    transform = from_origin(minx, maxy, res, res)

    os.makedirs(os.path.dirname(os.path.abspath(output_raster)), exist_ok=True)

    with rasterio.open(
        output_raster,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(pred_grid, 1)

    prog(100)
    stats = {
        "min": float(np.nanmin(predictions)),
        "max": float(np.nanmax(predictions)),
        "mean": float(np.nanmean(predictions)),
    }
    log(f"\n✓ 래스터 저장: {output_raster}")
    log(f"   통계: min={stats['min']:.4f}  max={stats['max']:.4f}  mean={stats['mean']:.4f}")
    return output_raster
