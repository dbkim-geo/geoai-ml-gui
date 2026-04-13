"""
지리공간 전처리 핵심 모듈.

포인트 + 래스터 → 버퍼(Circle/Moore) → Zonal Statistics → CSV
예측용: extent 기반 가상 격자 포인트 생성 후 동일 처리
"""
from __future__ import annotations

import json
import os
import traceback
import warnings
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point, box

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def get_raster_info(path: str) -> dict:
    """래스터 기본 정보 반환."""
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "res_x": src.res[0],
            "res_y": src.res[1],
            "bounds": src.bounds,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
            "width": src.width,
            "height": src.height,
        }


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _make_circle_buffers(gdf: gpd.GeoDataFrame, size: float) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry"] = gdf.geometry.buffer(size)
    return out


def _make_moore_buffers(gdf: gpd.GeoDataFrame, size: float) -> gpd.GeoDataFrame:
    """Square (Moore neighbourhood / bbox) buffer."""
    out = gdf.copy()
    out["geometry"] = gdf.geometry.apply(
        lambda p: box(p.x - size, p.y - size, p.x + size, p.y + size)
    )
    return out


def _make_buffers(gdf: gpd.GeoDataFrame, size: float, method: str) -> gpd.GeoDataFrame:
    if method == "circle":
        return _make_circle_buffers(gdf, size)
    return _make_moore_buffers(gdf, size)


# ---------------------------------------------------------------------------
# Zonal statistics helpers
# ---------------------------------------------------------------------------

def _zonal_continuous(
    buf_gdf: gpd.GeoDataFrame, raster_path: str, prefix: str, nodata
) -> pd.DataFrame:
    """연속형 래스터 → zonal mean."""
    stats = zonal_stats(
        list(buf_gdf.geometry),
        raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=True,
    )
    return pd.DataFrame({f"{prefix}_mean": [s.get("mean") for s in stats]})


def _zonal_categorical(
    buf_gdf: gpd.GeoDataFrame, raster_path: str, prefix: str, nodata
) -> pd.DataFrame:
    """범주형 래스터 → 클래스별 픽셀 count."""
    stats = zonal_stats(
        list(buf_gdf.geometry),
        raster_path,
        categorical=True,
        nodata=nodata,
        all_touched=True,
    )
    # Gather all classes: None 키와 nodata 값 제외
    all_classes: list = sorted({
        k for s in stats for k in s.keys()
        if k is not None and (nodata is None or k != nodata)
    })
    if not all_classes:
        return pd.DataFrame()
    rows = []
    for s in stats:
        row = {}
        for c in all_classes:
            try:
                col_name = f"{prefix}_cls{int(c)}_cnt"
            except (ValueError, TypeError):
                col_name = f"{prefix}_cls{c}_cnt"
            row[col_name] = s.get(c, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training data preprocessing
# ---------------------------------------------------------------------------

def preprocess_training(
    point_path: str,
    raster_configs: list[dict],
    buffer_sizes: list[float],
    buffer_method: str,
    target_col: str,
    output_csv: str,
    output_vector: Optional[str] = None,   # SHP / GPKG 경로 (None이면 생략)
    progress_cb: Optional[Callable] = None,
    log_cb: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    학습 데이터 전처리.

    Parameters
    ----------
    raster_configs : list of dicts
        {'path': str, 'name': str, 'type': 'continuous'|'categorical'}
    buffer_sizes : list of floats (CRS 단위, 보통 metre)
    buffer_method : 'circle' | 'moore'
    target_col : str  – 포인트 파일 내 종속변수 컬럼명

    Returns
    -------
    pd.DataFrame
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def prog(v: float):
        if progress_cb:
            progress_cb(int(v))

    # --- Load points ---
    log(f"포인트 파일 로드: {point_path}")
    gdf = gpd.read_file(point_path)
    log(f"  → {len(gdf)}개 포인트  |  CRS: {gdf.crs}")

    if not raster_configs:
        raise ValueError("래스터 파일을 하나 이상 추가하세요.")

    # --- Reference CRS from first raster ---
    ref_info = get_raster_info(raster_configs[0]["path"])
    ref_crs = ref_info["crs"]
    log(f"기준 CRS (첫 번째 래스터): {ref_crs}")

    if gdf.crs is None:
        raise ValueError("포인트 파일에 CRS 정보가 없습니다.")
    if gdf.crs != ref_crs:
        log(f"좌표 변환: {gdf.crs} → {ref_crs}")
        gdf = gdf.to_crs(ref_crs)

    # --- Build result DataFrame ---
    result = pd.DataFrame({"fid": range(len(gdf))})
    if target_col and target_col in gdf.columns:
        result[target_col] = gdf[target_col].values
        log(f"종속변수 컬럼 포함: '{target_col}'")
    else:
        log(f"[경고] 종속변수 컬럼 '{target_col}'을 포인트 파일에서 찾을 수 없습니다.")

    total = len(buffer_sizes) * len(raster_configs)
    step = 0

    for buf_size in buffer_sizes:
        log(f"\n--- 버퍼 {buf_size}m ({buffer_method}) ---")
        buf_gdf = _make_buffers(gdf, buf_size, buffer_method)

        for cfg in raster_configs:
            rname = cfg["name"]
            rtype = cfg["type"]
            rpath = cfg["path"]
            prefix = f"{rname}_{int(buf_size)}m"

            log(f"  {rname} ({'연속형' if rtype == 'continuous' else '범주형'})")
            rinfo = get_raster_info(rpath)

            try:
                if rtype == "continuous":
                    sdf = _zonal_continuous(buf_gdf, rpath, prefix, rinfo["nodata"])
                else:
                    sdf = _zonal_categorical(buf_gdf, rpath, prefix, rinfo["nodata"])

                for col in sdf.columns:
                    result[col] = sdf[col].values

            except Exception as exc:
                log(f"  [오류] {rname}: {exc}")
                log(traceback.format_exc())

            step += 1
            prog(step / total * 100)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log(f"\n✓ CSV 저장: {output_csv}")
    log(f"   {len(result)}행 × {len(result.columns)}열")

    # 벡터 파일 저장 (SHP / GPKG)
    if output_vector:
        _save_result_as_vector(gdf, result, output_vector, log)

    return result


def _save_result_as_vector(
    point_gdf: gpd.GeoDataFrame,
    result_df: pd.DataFrame,
    output_vector: str,
    log_fn,
):
    """전처리 결과를 포인트 지오메트리와 결합해 벡터 파일로 저장."""
    ext = os.path.splitext(output_vector)[1].lower()
    driver = "GPKG" if ext == ".gpkg" else "ESRI Shapefile"

    # 결과 컬럼을 GeoDataFrame에 붙이기
    out_gdf = point_gdf[["geometry"]].copy().reset_index(drop=True)
    for col in result_df.columns:
        out_gdf[col] = result_df[col].values

    os.makedirs(os.path.dirname(os.path.abspath(output_vector)), exist_ok=True)

    if driver == "ESRI Shapefile":
        # Shapefile 필드명 10자 제한 → 자동 단축 후 매핑 저장
        rename_map = {}
        used = set()
        new_cols = {}
        for col in out_gdf.columns:
            if col == "geometry":
                continue
            short = col[:10]
            idx = 1
            while short in used:
                suffix = str(idx)
                short = col[: 10 - len(suffix)] + suffix
                idx += 1
            used.add(short)
            rename_map[col] = short
            new_cols[short] = col

        out_gdf = out_gdf.rename(columns=rename_map)

        # 매핑 CSV 저장
        map_path = output_vector.replace(".shp", "_field_map.csv")
        pd.DataFrame(
            [{"short": k, "original": v} for k, v in new_cols.items()]
        ).to_csv(map_path, index=False, encoding="utf-8-sig")
        log_fn(f"   필드명 매핑: {map_path}")

    try:
        out_gdf.to_file(output_vector, driver=driver, encoding="utf-8")
        log_fn(f"✓ 벡터 저장: {output_vector}")
    except Exception as exc:
        log_fn(f"[경고] 벡터 저장 실패: {exc}")


# ---------------------------------------------------------------------------
# Prediction data preprocessing
# ---------------------------------------------------------------------------

def preprocess_prediction(
    raster_configs: list[dict],
    buffer_sizes: list[float],
    buffer_method: str,
    extent: tuple[float, float, float, float],  # (minx, miny, maxx, maxy)
    resolution: Optional[float],
    output_csv: str,
    progress_cb: Optional[Callable] = None,
    log_cb: Optional[Callable] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    예측 데이터 전처리.

    격자 포인트 생성 → 버퍼 → Zonal statistics → CSV + grid_meta JSON

    Returns
    -------
    (DataFrame, grid_meta dict)
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def prog(v: float):
        if progress_cb:
            progress_cb(int(v))

    if not raster_configs:
        raise ValueError("래스터 파일을 하나 이상 추가하세요.")

    ref_info = get_raster_info(raster_configs[0]["path"])
    ref_crs = ref_info["crs"]

    if resolution is None or resolution <= 0:
        resolution = ref_info["res_x"]
        log(f"해상도: 래스터 자동 감지 → {resolution}m")
    else:
        log(f"해상도: {resolution}m")

    minx, miny, maxx, maxy = extent
    log(f"기준 CRS: {ref_crs}")
    log(f"Extent: ({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f})")

    # --- Create grid points (cell centers) ---
    xs = np.arange(minx + resolution / 2, maxx, resolution)
    ys = np.arange(miny + resolution / 2, maxy, resolution)
    nx, ny = len(xs), len(ys)
    log(f"격자: {nx} × {ny} = {nx * ny:,}개 포인트")

    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.flatten(), gy.flatten()

    points = [Point(x, y) for x, y in zip(gx, gy)]
    point_gdf = gpd.GeoDataFrame(geometry=points, crs=ref_crs)

    result = pd.DataFrame({"grid_x": gx, "grid_y": gy})

    total = len(buffer_sizes) * len(raster_configs)
    step = 0

    for buf_size in buffer_sizes:
        log(f"\n--- 버퍼 {buf_size}m ({buffer_method}) ---")
        buf_gdf = _make_buffers(point_gdf, buf_size, buffer_method)

        for cfg in raster_configs:
            rname = cfg["name"]
            rtype = cfg["type"]
            rpath = cfg["path"]
            prefix = f"{rname}_{int(buf_size)}m"

            log(f"  {rname} ({'연속형' if rtype == 'continuous' else '범주형'})")
            rinfo = get_raster_info(rpath)

            try:
                if rtype == "continuous":
                    sdf = _zonal_continuous(buf_gdf, rpath, prefix, rinfo["nodata"])
                else:
                    sdf = _zonal_categorical(buf_gdf, rpath, prefix, rinfo["nodata"])

                for col in sdf.columns:
                    result[col] = sdf[col].values

            except Exception as exc:
                log(f"  [오류] {rname}: {exc}")
                log(traceback.format_exc())

            step += 1
            prog(step / total * 100)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    grid_meta = {
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "resolution": resolution,
        "nx": nx,
        "ny": ny,
        "crs": str(ref_crs),
    }
    meta_path = output_csv.replace(".csv", "_grid_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(grid_meta, f, indent=2, ensure_ascii=False)

    log(f"\n✓ 저장 완료: {output_csv}")
    log(f"   격자 메타데이터: {meta_path}")
    log(f"   {len(result):,}행 × {len(result.columns)}열")

    return result, grid_meta
