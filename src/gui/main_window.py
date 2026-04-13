"""
메인 윈도우 – QTabWidget으로 4개 탭 조립.
탭 간 데이터 연결(signal routing) 처리.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from src.gui.tab_predict import PredictTab
from src.gui.tab_pred_preprocess import PredPreprocessTab
from src.gui.tab_preprocess import PreprocessTab
from src.gui.tab_train import TrainTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoAI ML GUI  –  지리공간 머신러닝 실험 도구")
        self.resize(1280, 900)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Tabs ──────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setFont(QFont("Malgun Gothic", 10))

        self._tab1 = PreprocessTab()
        self._tab2 = TrainTab()
        self._tab3 = PredPreprocessTab()
        self._tab4 = PredictTab()

        self._tabs.addTab(self._tab1, "1. 학습데이터 전처리")
        self._tabs.addTab(self._tab2, "2. 모델 학습")
        self._tabs.addTab(self._tab3, "3. 예측데이터 전처리")
        self._tabs.addTab(self._tab4, "4. 예측")

        self.setCentralWidget(self._tabs)

        # ── Menu ──────────────────────────────────────────────────────
        menu = self.menuBar()
        help_menu = menu.addMenu("도움말")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

        # ── Status bar ────────────────────────────────────────────────
        sb = QStatusBar()
        sb.addPermanentWidget(QLabel("GeoAI ML GUI  |  준비"))
        self.setStatusBar(sb)

    def _connect_signals(self):
        # Tab1 → Tab2: 전처리 CSV 자동 입력
        self._tab1.output_path_changed.connect(self._tab2.set_input_csv)

        # Tab2 → Tab4: 학습된 모델 자동 입력
        self._tab2.model_trained.connect(self._tab4.add_model)

        # Tab3 → Tab4: 예측 CSV + grid_meta 자동 입력
        self._tab3.output_path_changed.connect(self._tab4.set_pred_csv)

    # ------------------------------------------------------------------
    def _show_about(self):
        QMessageBox.about(
            self,
            "About",
            "<b>GeoAI ML GUI</b><br><br>"
            "지리공간 데이터 기반 머신러닝 실험 도구<br><br>"
            "<b>주요 기능</b><br>"
            "• 포인트 + 래스터 → Zonal Statistics (연속형/범주형)<br>"
            "• 버퍼 방법: 원형(Circle) / 격자(Moore/BBox)<br>"
            "• 10+ ML 모델 학습 및 하이퍼파라미터 튜닝<br>"
            "• 예측 격자 생성 → GeoTIFF 출력<br>"
            "• SHAP 분석 (학습/예측 각각 저장)<br><br>"
            "Libraries: geopandas · rasterio · rasterstats · scikit-learn<br>"
            "xgboost · lightgbm · catboost · shap · matplotlib",
        )
