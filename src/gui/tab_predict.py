"""
Tab 4 – 예측.

입력: 예측 CSV + 학습된 모델 pkl + (자동감지) grid_meta
처리: predict → GeoTIFF 저장, (선택) SHAP 분석
"""
from __future__ import annotations

import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.prediction import load_grid_meta
from src.gui.widgets.mpl_canvas import MplCanvas
from src.workers.predict_worker import PredictWorker


class PredictTab(QWidget):
    """Tab 4 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PredictWorker | None = None
        self._grid_meta: dict | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Settings panel
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.addWidget(self._build_input_group())
        tl.addWidget(self._build_output_group())

        scroll = QScrollArea()
        scroll.setWidget(top)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(260)

        # Bottom: progress + log + chart
        bottom = QSplitter(Qt.Horizontal)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        left_l.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono())
        left_l.addWidget(self._log)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.addWidget(QLabel("SHAP 결과 미리보기"))
        self._canvas = MplCanvas(width=8, height=5)
        right_l.addWidget(self._canvas)

        bottom.addWidget(left_w)
        bottom.addWidget(right_w)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)

        splitter.addWidget(scroll)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    # --- Input group ---
    def _build_input_group(self) -> QGroupBox:
        gb = QGroupBox("입력")
        lay = QVBoxLayout(gb)

        # CSV
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("예측 CSV:"))
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText("예측 전처리 완료 CSV")
        self._csv_edit.textChanged.connect(self._on_csv_changed)
        row1.addWidget(self._csv_edit)
        btn1 = QPushButton("찾기")
        btn1.clicked.connect(self._browse_csv)
        row1.addWidget(btn1)
        lay.addLayout(row1)

        # Grid meta info
        self._meta_label = QLabel("※ grid_meta JSON: 자동 감지 대기 중")
        self._meta_label.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(self._meta_label)

        # Model
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("학습 모델 (.pkl):"))
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("joblib으로 저장된 모델 파일")
        row2.addWidget(self._model_edit)
        btn2 = QPushButton("찾기")
        btn2.clicked.connect(self._browse_model)
        row2.addWidget(btn2)
        lay.addLayout(row2)

        # SHAP option
        row3 = QHBoxLayout()
        self._chk_shap = QCheckBox("예측 데이터 SHAP 분석 수행 (시간 소요)")
        row3.addWidget(self._chk_shap)
        row3.addStretch()
        lay.addLayout(row3)

        return gb

    # --- Output group ---
    def _build_output_group(self) -> QGroupBox:
        gb = QGroupBox("출력")
        lay = QVBoxLayout(gb)

        row = QHBoxLayout()
        row.addWidget(QLabel("출력 래스터 (.tif):"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("prediction_map.tif")
        row.addWidget(self._out_edit)
        btn = QPushButton("찾기")
        btn.clicked.connect(self._browse_output)
        row.addWidget(btn)
        lay.addLayout(row)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  예측 실행")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setStyleSheet("font-weight: bold; background: #9C27B0; color: white;")
        self._btn_run.clicked.connect(self._run)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        lay.addLayout(btn_row)

        return gb

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "예측 CSV 선택", "", "CSV (*.csv)")
        if path:
            self._csv_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", "", "모델 (*.pkl *.joblib);;모든 파일 (*.*)"
        )
        if path:
            self._model_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 래스터 저장", "", "GeoTIFF (*.tif *.tiff)"
        )
        if path:
            if not (path.endswith(".tif") or path.endswith(".tiff")):
                path += ".tif"
            self._out_edit.setText(path)

    def _on_csv_changed(self, text: str):
        """CSV 변경 시 grid_meta 자동 감지."""
        path = text.strip()
        if not path:
            return
        meta = load_grid_meta(path)
        if meta:
            self._grid_meta = meta
            self._meta_label.setText(
                f"✓ grid_meta 감지: {meta['nx']}×{meta['ny']} 격자, "
                f"해상도={meta['resolution']}m, CRS={meta['crs']}"
            )
            self._meta_label.setStyleSheet("color: green; font-size: 10px;")
        else:
            self._grid_meta = None
            self._meta_label.setText("※ grid_meta JSON 없음 – 예측 전처리 CSV를 사용하세요.")
            self._meta_label.setStyleSheet("color: orange; font-size: 10px;")

    def _run(self):
        csv = self._csv_edit.text().strip()
        model_path = self._model_edit.text().strip()
        out_raster = self._out_edit.text().strip()

        if not csv:
            QMessageBox.warning(self, "입력 오류", "예측 CSV를 선택하세요.")
            return
        if not model_path:
            QMessageBox.warning(self, "입력 오류", "학습 모델 파일을 선택하세요.")
            return
        if not out_raster:
            QMessageBox.warning(self, "입력 오류", "출력 래스터 경로를 입력하세요.")
            return
        if self._grid_meta is None:
            meta = load_grid_meta(csv)
            if meta is None:
                QMessageBox.critical(
                    self, "오류",
                    "격자 메타데이터(grid_meta JSON)를 찾을 수 없습니다.\n"
                    "Tab3에서 예측 전처리를 먼저 수행하세요."
                )
                return
            self._grid_meta = meta

        params = {
            "csv_path": csv,
            "model_path": model_path,
            "output_raster": out_raster,
            "grid_meta": self._grid_meta,
            "run_shap": self._chk_shap.isChecked(),
        }

        self._log.clear()
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)

        self._worker = PredictWorker(params, parent=self)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, raster_path: str):
        self._progress.setValue(100)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[완료] 래스터: {raster_path}")
        QMessageBox.information(self, "완료", f"예측 완료!\n{raster_path}")

        # Show SHAP if available
        if self._chk_shap.isChecked():
            shap_dir = os.path.join(os.path.dirname(raster_path), "shap", "predict")
            if os.path.isdir(shap_dir):
                pngs = [
                    os.path.join(shap_dir, f)
                    for f in os.listdir(shap_dir)
                    if f.endswith(".png")
                ]
                if pngs:
                    self._canvas.show_image(sorted(pngs)[0])
                    self._log.append(f"SHAP 이미지: {shap_dir}")

    def _on_error(self, msg: str):
        self._progress.setValue(0)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[오류]\n{msg}")
        QMessageBox.critical(self, "오류", msg[:500])

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------
    def set_pred_csv(self, path: str, grid_meta: dict):
        self._csv_edit.setText(path)
        self._grid_meta = grid_meta
        self._meta_label.setText(
            f"✓ grid_meta (Tab3): {grid_meta['nx']}×{grid_meta['ny']} 격자"
        )
        self._meta_label.setStyleSheet("color: green; font-size: 10px;")

    def add_model(self, results: dict):
        """Tab2에서 학습 완료 시 첫 번째 모델 경로 자동 입력."""
        for name, res in results.items():
            if res.get("model_path"):
                self._model_edit.setText(res["model_path"])
                self._log.append(f"[자동] 모델 설정: {res['model_path']}")
                break


def _mono():
    f = QFont("Consolas")
    f.setPointSize(9)
    return f
