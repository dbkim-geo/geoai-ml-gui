# GeoAI ML GUI

지리공간(Geospatial) 데이터를 활용한 머신러닝 실험 도구입니다.  
포인트·래스터 데이터를 입력받아 버퍼 기반 공간 통계를 추출하고,  
10개 이상의 ML 모델을 학습·비교한 뒤 예측 래스터 지도(GeoTIFF)를 출력합니다.

---

## 주요 기능

| 탭 | 기능 |
|---|---|
| **1. 학습데이터 전처리** | 포인트 + 다중 래스터 → 버퍼(원형/격자) → Zonal Statistics → CSV |
| **2. 모델 학습** | 10+ ML 모델 선택·학습 / 하이퍼파라미터 튜닝 / 거리별 모델링 / SHAP 분석 |
| **3. 예측데이터 전처리** | Extent 기준 가상 격자 포인트 자동 생성 → 동일 Zonal Statistics → CSV |
| **4. 예측** | 학습 모델로 예측 → GeoTIFF 출력 / 예측 데이터 SHAP 분석 |

---

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/dbkim-geo/geoai-ml-gui.git
cd geoai-ml-gui
```

### 2. 가상환경 활성화 (Windows)
```bash
geoai-ml-gui\Scripts\activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

> **참고**: XGBoost / LightGBM / CatBoost는 선택적 라이브러리입니다.  
> 설치된 경우 자동으로 모델 목록에 추가됩니다.

### 4. 실행
```bash
python main.py
```

---

## 사용 방법

### Tab 1 – 학습데이터 전처리

1. **포인트 파일** (SHP / GPKG / GeoJSON) 선택
   - 파일 로드 시 컬럼 목록이 자동으로 드롭다운에 채워집니다.
2. **종속변수 컬럼** 선택 (드롭다운)
3. **작업 유형** 선택: 회귀(Regression) / 분류(Classification)
4. **래스터 파일** 추가 (+ 버튼), 각 래스터마다 **연속형 / 범주형** 선택
   - 연속형: Zonal Statistics **Mean** 계산
   - 범주형: Zonal Histogram (클래스별 **Count**) 계산
5. **버퍼 방법** 선택
   - 원형 (Circle): `buffer(r)`
   - 격자 (Moore / BBox): 정사각형 버퍼 `box(cx±r, cy±r)`
6. **버퍼 크기** 범위 입력 (최소 / 최대 / 간격, 단위: m)
   - 예) 최소=100, 최대=500, 간격=100 → 100m, 200m, 300m, 400m, 500m 5개 버퍼
7. **출력 CSV 경로** 지정 → **전처리 실행**

**출력 컬럼 예시**
```
fid, target, slope_100m_mean, slope_200m_mean,
              landcover_100m_cls1_cnt, landcover_100m_cls2_cnt, ...
```

---

### Tab 2 – 모델 학습

1. **학습 CSV** 선택 (Tab1 출력, 자동 연결)
   - CSV 로드 시 컬럼 목록이 자동으로 드롭다운에 채워집니다.
2. **종속변수 컬럼** 선택 (드롭다운)
3. **모델 선택** 체크박스 (전체 선택/해제 버튼 제공)
4. **학습 옵션**: 하이퍼파라미터 튜닝 / 랜덤서치 횟수 / CV Fold / 테스트 비율 / SHAP 분석
5. **거리별 모델링** (선택)
   - **비활성**: 전체 버퍼 거리 통합 학습 (기본)
   - **개별 거리별 모델**: 각 버퍼 거리마다 독립적인 모델 학습 및 성능 비교
   - **변수별 최적 거리 자동 선택**: 각 변수마다 타겟과 상관성이 가장 높은 거리를 선택 후 단일 모델 학습
     - 회귀: Spearman 상관계수 기준
     - 분류: Mutual Information 기준
6. **출력 디렉토리** 지정 → **학습 실행**

**출력 파일 구조**
```
output_dir/
├── ModelName.pkl              # 학습된 모델 (joblib)
├── metrics_summary.csv        # 모델별 성능 요약
├── charts/
│   ├── performance_comparison.png
│   ├── actual_vs_predicted.png    (회귀 전용)
│   └── ModelName_feature_importance.png
└── shap/train/
    ├── ModelName_train_shap_summary.png
    └── ModelName_train_shap_importance.png

# 거리별 모드일 때
output_dir/
├── 100m/
│   ├── ModelName.pkl
│   ├── metrics_summary.csv
│   └── charts/
├── 200m/ ...
├── scale_opt/                       # 변수별 최적 스케일 선택 모드 (Scale of Effect)
│   ├── ModelName.pkl
│   └── charts/
├── scale_opt_selection.csv          # 변수별 선택된 최적 거리 결과 (Scale Optimization)
└── distance_comparison.png      # 거리별 성능 비교 차트
```

---

### Tab 3 – 예측데이터 전처리

1. **예측 공간 범위(Extent)** 설정
   - 래스터에서 자동 감지 (체크박스)
   - 또는 MinX / MinY / MaxX / MaxY 수동 입력
2. **해상도** 설정 (자동 감지 또는 수동 입력, 단위: m)
3. 래스터 파일 및 버퍼 설정은 **Tab1과 동일하게** 입력
4. **출력 CSV 경로** 지정 → **예측 전처리 실행**

> 예측 CSV와 함께 `_grid_meta.json` (격자 메타데이터) 파일이 자동 생성됩니다.  
> Tab4에서 이 JSON을 자동으로 읽어 예측 래스터를 복원합니다.

---

### Tab 4 – 예측

1. **예측 CSV** 선택 (Tab3 출력, 자동 연결)
   - `_grid_meta.json` 자동 감지 및 표시
2. **학습 모델 (.pkl)** 선택 (Tab2 완료 시 자동 입력)
3. **출력 래스터 경로** (.tif) 지정
4. **SHAP 분석** 옵션 체크 (선택)
5. **예측 실행**

> 예측 결과는 GeoTIFF 형식으로 저장되며,  
> SHAP 분석 결과는 `output_dir/shap/predict/` 에 저장됩니다.

---

## 지원 모델

| 모델 | 회귀 | 분류 | 비고 |
|---|:---:|:---:|---|
| Random Forest | ✅ | ✅ | |
| Extra Trees | ✅ | ✅ | |
| Gradient Boosting | ✅ | ✅ | |
| AdaBoost | ✅ | ✅ | |
| Decision Tree | ✅ | ✅ | |
| SVR / SVC | ✅ | ✅ | |
| KNN | ✅ | ✅ | |
| Ridge | ✅ | — | |
| Lasso | ✅ | — | |
| ElasticNet | ✅ | — | |
| Logistic Regression | — | ✅ | |
| MLP (Neural Net) | ✅ | ✅ | |
| **XGBoost** | ✅ | ✅ | 설치 시 활성화 |
| **LightGBM** | ✅ | ✅ | 설치 시 활성화 |
| **CatBoost** | ✅ | ✅ | 설치 시 활성화 |

---

## 프로젝트 구조

```
geoai-ml-gui/
├── main.py                         # 앱 진입점
├── requirements.txt
└── src/
    ├── core/
    │   ├── preprocessing.py        # Zonal statistics (연속형 mean / 범주형 count)
    │   ├── training.py             # ML 학습, 거리별 모델링, 차트 생성
    │   ├── prediction.py           # 예측 → GeoTIFF 저장
    │   └── shap_utils.py           # SHAP 분석 (train / predict 구분)
    ├── workers/
    │   ├── preprocess_worker.py    # QThread 전처리 (학습용 / 예측용)
    │   ├── train_worker.py         # QThread 학습 (standard / per_distance / 변수별 최적 거리)
    │   └── predict_worker.py       # QThread 예측 + SHAP
    └── gui/
        ├── main_window.py          # 메인 윈도우 + 탭 간 신호 연결
        ├── tab_preprocess.py       # Tab1
        ├── tab_train.py            # Tab2
        ├── tab_pred_preprocess.py  # Tab3
        ├── tab_predict.py          # Tab4
        └── widgets/
            ├── raster_table.py     # 래스터 목록 위젯 (연속형/범주형 선택)
            └── mpl_canvas.py       # matplotlib 임베드 캔버스
```

---

## 의존성

```
PyQt5 · geopandas · rasterio · rasterstats · numpy · pandas
scikit-learn · matplotlib · shapely · pyproj · scipy · joblib · shap
(optional) xgboost · lightgbm · catboost
```

---

## 인용 / Citation

이 도구를 연구에 활용하신다면, 아래 관련 논문을 인용해 주시기 바랍니다.

If you use this tool in your research, please cite the relevant papers below.

---

Lee, G., Cho, Y., Han, Y., & Kim, G. (2025). Characterizing the relationship between the spatial range of influence of urban land characteristics and surface temperature using geospatial explainable artificial intelligence models. *International Journal of Digital Earth*, 18(2), 2583833.

Kim, G., Cho, Y., Lee, J. H., & Lee, G. (2025). Correlation analysis between urban environment features and crime occurrence based on explainable artificial intelligence techniques. *Journal of Asian Architecture and Building Engineering*, 24(6), 5751–5770.

Ha, J., Lee, J., Lee, G., & Kim, G. (2025). A methodology to support species selection decisions for planting trees in urban spaces using explainable AI. *Environmental Research Communications*, 7(11), 115019.

Kim, G., Cho, Y., Han, Y., & Lee, G. (2025). Crime mapping in urban environments using explainable AI: A case study of Daegu, Korea. *Sustainable Cities and Society*, 130, 106507.

Gunwon, L., Yuhan, H., & Kim, G. (2025). Impact of land use characteristics on air pollutant concentrations considering the spatial range of influence. *Atmospheric Pollution Research*, 16(6), 102498.

김근한. (2023). GeoXAI를 활용한 서울시 탄소흡수 예측지도 제작. *한국기후변화학회지*, 14(6-1), 871–879.

이준우, 한유한, 이정택, 박진혁, & 김근한. (2023). 식생지수를 활용한 LULUCF 정주지 온실가스인벤토리 산정을 위한 수목탐지 방법 개발. *Korean Journal of Remote Sensing*, 39(6-3), 1721–1730.

Kim, M., Kim, D., Jin, D., & Kim, G. (2023). Application of explainable artificial intelligence (XAI) in urban growth modeling: A case study of Seoul metropolitan area, Korea. *Land*, 12(2), 420.

Kim, M., Kim, D., & Kim, G. (2022). Examining the relationship between land use/land cover (LULC) and land surface temperature (LST) using explainable artificial intelligence (XAI) models: A case study of Seoul, South Korea. *International Journal of Environmental Research and Public Health*, 19(23), 15926.

Kim, M., & Kim, G. (2022). Modeling and predicting urban expansion in South Korea using explainable artificial intelligence (XAI) model. *Applied Sciences*, 12(18), 9169.
