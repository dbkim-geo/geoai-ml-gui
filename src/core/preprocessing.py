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
from pyproj import CRS as _ProjCRS, Transformer
from scipy.ndimage import convolve as _nd_convolve
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


def _reproject_to_raster_crs(buf_gdf: gpd.GeoDataFrame, raster_path: str) -> gpd.GeoDataFrame:
    """버퍼 GDF를 래스터 CRS에 맞게 재투영. CRS가 같으면 그대로 반환."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    if buf_gdf.crs is None:
        return buf_gdf
    if buf_gdf.crs == raster_crs:
        return buf_gdf
    return buf_gdf.to_crs(raster_crs)


# ---------------------------------------------------------------------------
# Fast focal statistics (예측 격자용 — 래스터 한 번 읽고 convolution)
# ---------------------------------------------------------------------------

def _make_focal_kernel(buffer_size: float, res: float, method: str) -> np.ndarray:
    """
    버퍼 방식에 따라 focal statistics 커널 생성.

    circle : x²+y² ≤ r² 인 원형 커널
    moore  : 2r+1 × 2r+1 정사각형 커널
    """
    r_px = max(1, int(np.ceil(buffer_size / res)))
    if method == "circle":
        yy, xx = np.ogrid[-r_px : r_px + 1, -r_px : r_px + 1]
        kernel = (xx ** 2 + yy ** 2 <= r_px ** 2).astype(np.float64)
    else:  # moore / bbox
        kernel = np.ones((2 * r_px + 1, 2 * r_px + 1), dtype=np.float64)
    return kernel


def _read_raster_window(raster_path: str, xs: np.ndarray, ys: np.ndarray,
                         src_crs, buffer_size: float):
    """
    예측 격자 범위 + 버퍼 패딩만큼의 래스터 윈도우를 읽는다.
    (xs, ys 는 src_crs 좌표계)

    Returns
    -------
    data        : 2D ndarray (float64)
    win_tf      : 윈도우 affine transform
    res         : 래스터 해상도
    nodata      : 래스터 nodata 값
    qx, qy      : 래스터 CRS로 변환된 쿼리 좌표
    """
    import rasterio.windows

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        res = src.res[0]

        # 좌표 CRS 변환
        sc = _ProjCRS.from_user_input(src_crs)
        rc = _ProjCRS.from_user_input(raster_crs)
        if sc != rc:
            tr = Transformer.from_crs(sc, rc, always_xy=True)
            qx, qy = np.array(tr.transform(xs, ys))
        else:
            qx, qy = np.asarray(xs, float), np.asarray(ys, float)

        # extent + 패딩만큼 윈도우 잘라 읽기
        pad = buffer_size + res * 2
        win = rasterio.windows.from_bounds(
            qx.min() - pad, qy.min() - pad,
            qx.max() + pad, qy.max() + pad,
            transform=src.transform,
        )
        # 래스터 전체 범위로 클리핑
        win = win.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        data = src.read(1, window=win).astype(np.float64)
        win_tf = src.window_transform(win)
        nodata = src.nodata

    return data, win_tf, res, nodata, qx, qy


def _focal_continuous(
    raster_path: str,
    xs: np.ndarray, ys: np.ndarray,
    src_crs,
    buffer_size: float,
    method: str,
    log_cb=None,
) -> np.ndarray:
    """
    연속형 래스터: focal mean(원형 or bbox 커널) → 격자 좌표에서 샘플링.

    rasterstats 방식 대비 수십 배 빠름.
    """
    data, win_tf, res, nodata, qx, qy = _read_raster_window(
        raster_path, xs, ys, src_crs, buffer_size
    )

    # nodata 마스킹
    if nodata is not None:
        data[data == nodata] = np.nan

    kernel = _make_focal_kernel(buffer_size, res, method)

    # NaN-aware focal mean: sum(values) / sum(valid_pixels)
    valid = (~np.isnan(data)).astype(np.float64)
    filled = np.where(np.isnan(data), 0.0, data)
    sum_v = _nd_convolve(filled, kernel, mode="constant", cval=0.0)
    sum_n = _nd_convolve(valid,  kernel, mode="constant", cval=0.0)
    focal = np.where(sum_n > 0, sum_v / sum_n, np.nan)

    # 격자 위치에서 샘플링
    rows, cols = rasterio.transform.rowcol(win_tf, qx, qy)
    rows, cols = np.asarray(rows), np.asarray(cols)
    h, w = data.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    out = np.full(len(xs), np.nan, dtype=np.float64)
    out[inside] = focal[rows[inside], cols[inside]]

    if log_cb:
        valid_n = int(np.sum(~np.isnan(out)))
        log_cb(f"    → 유효 결과: {valid_n}/{len(xs)}개")
        if valid_n == 0:
            log_cb("    [경고] 모든 결과가 NaN. 래스터 Extent·nodata 확인 요망.")
    return out


def _focal_categorical(
    raster_path: str,
    xs: np.ndarray, ys: np.ndarray,
    src_crs,
    buffer_size: float,
    method: str,
    log_cb=None,
) -> dict:
    """
    범주형 래스터: 클래스별 focal count(원형 or bbox 커널) → 격자 좌표에서 샘플링.

    Returns
    -------
    dict[int, np.ndarray]  — {class_value: count_array}
    """
    data, win_tf, res, nodata, qx, qy = _read_raster_window(
        raster_path, xs, ys, src_crs, buffer_size
    )

    # valid 마스크 (nodata 제외)
    if nodata is not None:
        try:
            nd_val = data.dtype.type(nodata)
            valid_mask = data != nd_val
        except Exception:
            valid_mask = np.ones(data.shape, dtype=bool)
    else:
        valid_mask = np.ones(data.shape, dtype=bool)

    classes = sorted(np.unique(data[valid_mask]).tolist())
    if not classes:
        if log_cb:
            log_cb("    [경고] 유효 픽셀이 없습니다. CRS·Extent·nodata 확인 요망.")
        return {}
    if log_cb:
        log_cb(
            f"    → 클래스 {len(classes)}종: {[int(c) for c in classes[:10]]}"
            + (" ..." if len(classes) > 10 else "")
        )

    kernel = _make_focal_kernel(buffer_size, res, method)

    rows, cols = rasterio.transform.rowcol(win_tf, qx, qy)
    rows, cols = np.asarray(rows), np.asarray(cols)
    h, w = data.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    result = {}
    for cls in classes:
        binary = ((data == cls) & valid_mask).astype(np.float64)
        count_map = _nd_convolve(binary, kernel, mode="constant", cval=0.0)
        counts = np.zeros(len(xs), dtype=np.int32)
        counts[inside] = count_map[rows[inside], cols[inside]].astype(np.int32)
        result[int(cls)] = counts
    return result


# ---------------------------------------------------------------------------
# Zonal statistics helpers (학습 데이터용 — 불규칙 포인트)
# ---------------------------------------------------------------------------

def _zonal_continuous(
    buf_gdf: gpd.GeoDataFrame, raster_path: str, prefix: str, nodata,
    log_cb=None,
) -> pd.DataFrame:
    """연속형 래스터 → zonal mean."""
    aligned = _reproject_to_raster_crs(buf_gdf, raster_path)
    stats = zonal_stats(
        list(aligned.geometry),
        raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=True,
    )
    valid = sum(1 for s in stats if s.get("mean") is not None)
    if log_cb:
        log_cb(f"    → 유효 결과: {valid}/{len(stats)}개")
        if valid == 0:
            log_cb(f"    [경고] 모든 결과가 NaN입니다. 래스터 Extent·nodata 설정을 확인하세요.")
    return pd.DataFrame({f"{prefix}_mean": [s.get("mean") for s in stats]})


def _zonal_categorical(
    buf_gdf: gpd.GeoDataFrame, raster_path: str, prefix: str, nodata,
    log_cb=None,
) -> pd.DataFrame:
    """범주형 래스터 → 클래스별 픽셀 count."""
    aligned = _reproject_to_raster_crs(buf_gdf, raster_path)
    stats = zonal_stats(
        list(aligned.geometry),
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
        non_empty = sum(1 for s in stats if s)
        if log_cb:
            log_cb(
                f"  [경고] '{prefix}' 범주형 결과가 비어있습니다 "
                f"(non-empty zones: {non_empty}/{len(stats)}). "
                f"래스터 CRS·Extent·nodata 설정을 확인하세요."
            )
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
    if log_cb:
        log_cb(f"    → 클래스 {len(all_classes)}종: {[int(c) if isinstance(c, (int, float)) else c for c in all_classes[:10]]}"
               + (" ..." if len(all_classes) > 10 else ""))
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

            rinfo = get_raster_info(rpath)
            raster_crs = rinfo["crs"]
            crs_note = "" if buf_gdf.crs == raster_crs else f" [CRS 재투영: {buf_gdf.crs} → {raster_crs}]"
            log(f"  {rname} ({'연속형' if rtype == 'continuous' else '범주형'}){crs_note}")

            try:
                if rtype == "continuous":
                    sdf = _zonal_continuous(buf_gdf, rpath, prefix, rinfo["nodata"], log_cb=log)
                else:
                    sdf = _zonal_categorical(buf_gdf, rpath, prefix, rinfo["nodata"], log_cb=log)

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

    # --- Raster bounds vs. grid extent 사전 비교 ---
    log("\n[래스터 범위 확인]")
    for cfg in raster_configs:
        try:
            ri = get_raster_info(cfg["path"])
            rb = ri["bounds"]
            rcrs = ri["crs"]
            # 격자 extent를 래스터 CRS로 변환하여 비교
            from pyproj import CRS as ProjCRS
            grid_crs = ProjCRS.from_user_input(ref_crs)
            raster_crs_obj = ProjCRS.from_user_input(rcrs)
            if grid_crs != raster_crs_obj:
                transformer = Transformer.from_crs(grid_crs, raster_crs_obj, always_xy=True)
                gx0, gy0 = transformer.transform(minx, miny)
                gx1, gy1 = transformer.transform(maxx, maxy)
            else:
                gx0, gy0, gx1, gy1 = minx, miny, maxx, maxy
            overlap = (gx0 < rb.right and gx1 > rb.left and gy0 < rb.top and gy1 > rb.bottom)
            epsg = rcrs.to_epsg()
            crs_label = f"EPSG:{epsg}" if epsg else str(rcrs)[:20]
            log(
                f"  {cfg['name']} [{crs_label}]  "
                f"bounds=({rb.left:.0f},{rb.bottom:.0f},{rb.right:.0f},{rb.top:.0f})  "
                f"→ {'겹침 OK' if overlap else '[경고] 격자와 겹치지 않음!'}"
            )
        except Exception as e:
            log(f"  {cfg['name']} bounds 확인 실패: {e}")
    log("")

    # --- Create grid points (cell centers) ---
    xs = np.arange(minx + resolution / 2, maxx, resolution)
    ys = np.arange(miny + resolution / 2, maxy, resolution)
    nx, ny = len(xs), len(ys)
    log(f"격자: {nx} × {ny} = {nx * ny:,}개 포인트")

    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.flatten(), gy.flatten()

    result = pd.DataFrame({"grid_x": gx, "grid_y": gy})

    total = len(buffer_sizes) * len(raster_configs)
    step = 0

    log(f"[빠른 Focal Statistics 모드: 래스터 1회 읽기 + {'원형' if buffer_method == 'circle' else '격자'} 커널 convolution]")

    for buf_size in buffer_sizes:
        log(f"\n--- 버퍼 {buf_size}m ({buffer_method}) ---")

        for cfg in raster_configs:
            rname = cfg["name"]
            rtype = cfg["type"]
            rpath = cfg["path"]
            prefix = f"{rname}_{int(buf_size)}m"

            rinfo = get_raster_info(rpath)
            raster_crs_epsg = rinfo["crs"].to_epsg()
            grid_crs_epsg = ref_crs.to_epsg() if hasattr(ref_crs, "to_epsg") else None
            crs_note = (
                "" if raster_crs_epsg == grid_crs_epsg
                else f" [CRS 변환 → EPSG:{raster_crs_epsg}]"
            )
            log(f"  {rname} ({'연속형' if rtype == 'continuous' else '범주형'}){crs_note}")

            try:
                if rtype == "continuous":
                    means = _focal_continuous(
                        rpath, gx, gy, ref_crs, buf_size, buffer_method, log_cb=log
                    )
                    result[f"{prefix}_mean"] = means
                else:
                    cls_counts = _focal_categorical(
                        rpath, gx, gy, ref_crs, buf_size, buffer_method, log_cb=log
                    )
                    for cls, counts in cls_counts.items():
                        result[f"{prefix}_cls{cls}_cnt"] = counts

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
