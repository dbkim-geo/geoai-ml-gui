"""
Tab 1 – 학습 데이터 전처리.

입력: 포인트 shp + 다중 래스터(연속형/범주형)
처리: 좌표계 통일 → 버퍼(Circle/Moore) → Zonal stats → CSV
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import geopandas as gpd

from src.gui.widgets.raster_table import RasterTableWidget
from src.workers.preprocess_worker import TrainPreprocessWorker


class PreprocessTab(QWidget):
    """Tab 1 위젯."""

    output_path_changed = pyqtSignal(str)  # CSV 경로 → Tab2 전달용

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainPreprocessWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Top panel: settings
        top = QWidget()
        top_layout = QVBoxLayout(top)

        top_layout.addWidget(self._build_input_group())
        top_layout.addWidget(self._build_raster_group())
        top_layout.addWidget(self._build_buffer_group())
        top_layout.addWidget(self._build_output_group())

        # Scroll for top
        scroll = QScrollArea()
        scroll.setWidget(top)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(350)

        # Bottom panel: progress + log
        bottom = QWidget()
        bot_layout = QVBoxLayout(bottom)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        bot_layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono_font())
        bot_layout.addWidget(self._log)

        splitter.addWidget(scroll)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    # --- Input group ---
    def _build_input_group(self) -> QGroupBox:
        gb = QGroupBox("포인트 입력 (종속변수)")
        lay = QVBoxLayout(gb)

        # Point file
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("포인트 파일:"))
        self._point_edit = QLineEdit()
        self._point_edit.setPlaceholderText("SHP, GPKG, GeoJSON ...")
        row1.addWidget(self._point_edit)
        btn_pt = QPushButton("찾기")
        btn_pt.clicked.connect(self._browse_point)
        row1.addWidget(btn_pt)
        self._point_crs_label = QLabel("")
        self._point_crs_label.setStyleSheet("color: #1565C0; font-weight: bold;")
        row1.addWidget(self._point_crs_label)
        lay.addLayout(row1)

        # Target column + task type
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("종속변수 컬럼:"))
        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._target_combo.setMinimumWidth(180)
        self._target_combo.setPlaceholderText("파일 로드 후 선택")
        self._target_combo.lineEdit().setPlaceholderText("파일 로드 후 선택")
        row2.addWidget(self._target_combo)
        row2.addSpacing(20)

        row2.addWidget(QLabel("작업 유형:"))
        self._radio_reg = QRadioButton("회귀 (Regression)")
        self._radio_cls = QRadioButton("분류 (Classification)")
        self._radio_reg.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._radio_reg)
        grp.addButton(self._radio_cls)
        row2.addWidget(self._radio_reg)
        row2.addWidget(self._radio_cls)
        row2.addStretch()
        lay.addLayout(row2)

        return gb

    # --- Raster group ---
    def _build_raster_group(self) -> QGroupBox:
        gb = QGroupBox("독립변수 래스터")
        lay = QVBoxLayout(gb)
        self._raster_table = RasterTableWidget()
        self._raster_table.setMinimumHeight(160)
        lay.addWidget(self._raster_table)
        return gb

    # --- Buffer group ---
    def _build_buffer_group(self) -> QGroupBox:
        gb = QGroupBox("버퍼 설정")
        lay = QVBoxLayout(gb)

        # Method
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("버퍼 방법:"))
        self._radio_circle = QRadioButton("원형 (Circle)")
        self._radio_moore = QRadioButton("격자 (Moore / BBox)")
        self._radio_circle.setChecked(True)
        bgrp = QButtonGroup(self)
        bgrp.addButton(self._radio_circle)
        bgrp.addButton(self._radio_moore)
        row1.addWidget(self._radio_circle)
        row1.addWidget(self._radio_moore)
        row1.addStretch()
        lay.addLayout(row1)

        # Size range
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("크기 (m):  최소"))
        self._buf_min = QSpinBox()
        self._buf_min.setRange(1, 100000)
        self._buf_min.setValue(100)
        self._buf_min.setSingleStep(100)
        row2.addWidget(self._buf_min)

        row2.addWidget(QLabel("최대"))
        self._buf_max = QSpinBox()
        self._buf_max.setRange(1, 100000)
        self._buf_max.setValue(500)
        self._buf_max.setSingleStep(100)
        row2.addWidget(self._buf_max)

        row2.addWidget(QLabel("간격"))
        self._buf_step = QSpinBox()
        self._buf_step.setRange(1, 100000)
        self._buf_step.setValue(100)
        self._buf_step.setSingleStep(100)
        row2.addWidget(self._buf_step)

        row2.addStretch()
        lay.addLayout(row2)

        # Preview label
        self._buf_preview = QLabel()
        self._buf_min.valueChanged.connect(self._update_buf_preview)
        self._buf_max.valueChanged.connect(self._update_buf_preview)
        self._buf_step.valueChanged.connect(self._update_buf_preview)
        self._update_buf_preview()
        lay.addWidget(self._buf_preview)

        return gb

    # --- Output group ---
    def _build_output_group(self) -> QGroupBox:
        gb = QGroupBox("출력")
        lay = QVBoxLayout(gb)

        # CSV
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("출력 CSV:"))
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("train_data.csv")
        row1.addWidget(self._output_edit)
        btn_csv = QPushButton("찾기")
        btn_csv.clicked.connect(self._browse_output)
        row1.addWidget(btn_csv)
        lay.addLayout(row1)

        # Vector (SHP / GPKG)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("벡터 저장 (선택):"))
        self._vector_edit = QLineEdit()
        self._vector_edit.setPlaceholderText("train_points.gpkg  (SHP 또는 GPKG, 비워두면 생략)")
        row2.addWidget(self._vector_edit)
        btn_vec = QPushButton("찾기")
        btn_vec.clicked.connect(self._browse_vector)
        row2.addWidget(btn_vec)
        lay.addLayout(row2)

        note = QLabel("※ SHP 저장 시 필드명이 10자로 잘립니다. GPKG 권장.")
        note.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(note)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  전처리 실행")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setStyleSheet("font-weight: bold; background: #2196F3; color: white;")
        self._btn_run.clicked.connect(self._run)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        lay.addLayout(btn_row)

        return gb

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _browse_point(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "포인트 파일 선택", "",
            "벡터 파일 (*.shp *.gpkg *.geojson *.json);;모든 파일 (*.*)"
        )
        if path:
            self._point_edit.setText(path)
            self._load_point_columns(path)

    def _load_point_columns(self, path: str):
        """포인트 파일을 읽어 컬럼 목록과 CRS를 로드한다."""
        try:
            gdf = gpd.read_file(path, rows=1)
            # CRS 표시
            crs = gdf.crs
            if crs is not None:
                epsg = crs.to_epsg()
                crs_str = f"EPSG:{epsg}" if epsg else crs.to_string()[:30]
            else:
                crs_str = "CRS 없음"
            self._point_crs_label.setText(f"[{crs_str}]")

            cols = [c for c in gdf.columns if c != "geometry"]
            prev = self._target_combo.currentText()
            self._target_combo.clear()
            self._target_combo.addItems(cols)
            if prev in cols:
                self._target_combo.setCurrentText(prev)
            self._log.append(f"컬럼 로드 완료: {cols}  |  CRS: {crs_str}")
        except Exception as e:
            self._log.append(f"[경고] 컬럼 읽기 실패: {e}")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 CSV 저장", "", "CSV (*.csv)"
        )
        if path:
            if not path.endswith(".csv"):
                path += ".csv"
            self._output_edit.setText(path)

    def _browse_vector(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "벡터 파일 저장", "",
            "GeoPackage (*.gpkg);;Shapefile (*.shp)"
        )
        if path:
            self._vector_edit.setText(path)

    def _update_buf_preview(self):
        sizes = self._get_buffer_sizes()
        self._buf_preview.setText(f"  → 버퍼 크기 목록: {sizes} m")

    def _get_buffer_sizes(self) -> list[float]:
        bmin = self._buf_min.value()
        bmax = self._buf_max.value()
        bstep = self._buf_step.value()
        if bstep <= 0:
            return [bmin]
        return [float(v) for v in range(bmin, bmax + 1, bstep)]

    def _run(self):
        # Validate
        if not self._point_edit.text().strip():
            QMessageBox.warning(self, "입력 오류", "포인트 파일을 선택하세요.")
            return
        if self._raster_table.row_count() == 0:
            QMessageBox.warning(self, "입력 오류", "래스터 파일을 추가하세요.")
            return
        if not self._output_edit.text().strip():
            QMessageBox.warning(self, "입력 오류", "출력 CSV 경로를 입력하세요.")
            return

        params = {
            "point_path": self._point_edit.text().strip(),
            "raster_configs": self._raster_table.get_configs(),
            "buffer_sizes": self._get_buffer_sizes(),
            "buffer_method": "circle" if self._radio_circle.isChecked() else "moore",
            "target_col": self._target_combo.currentText().strip(),
            "output_csv": self._output_edit.text().strip(),
            "output_vector": self._vector_edit.text().strip() or None,
        }

        self._log.clear()
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)

        self._worker = TrainPreprocessWorker(params, parent=self)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _append_log(self, msg: str):
        self._log.append(msg)

    def _on_finished(self, csv_path: str):
        self._progress.setValue(100)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[완료] 출력: {csv_path}")
        QMessageBox.information(self, "완료", f"전처리 완료!\n{csv_path}")
        self.output_path_changed.emit(csv_path)

    def _on_error(self, msg: str):
        self._progress.setValue(0)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[오류]\n{msg}")
        QMessageBox.critical(self, "오류", msg[:500])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_task_type(self) -> str:
        return "regression" if self._radio_reg.isChecked() else "classification"


def _mono_font():
    from PyQt5.QtGui import QFont
    f = QFont("Consolas")
    f.setPointSize(9)
    return f
