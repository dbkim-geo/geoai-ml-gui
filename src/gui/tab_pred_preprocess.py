"""
Tab 3 – 예측 데이터 전처리.

입력: 래스터 + Extent (사용자 입력 or 래스터 자동 감지) + 버퍼 설정
처리: 가상 격자 포인트 생성 → Zonal stats → CSV + grid_meta JSON
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
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
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.preprocessing import get_raster_info
from src.gui.widgets.raster_table import RasterTableWidget
from src.workers.preprocess_worker import PredPreprocessWorker


class PredPreprocessTab(QWidget):
    """Tab 3 위젯."""

    output_path_changed = pyqtSignal(str, dict)  # (csv_path, grid_meta) → Tab4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PredPreprocessWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.addWidget(self._build_extent_group())
        tl.addWidget(self._build_raster_group())
        tl.addWidget(self._build_buffer_group())
        tl.addWidget(self._build_output_group())

        scroll = QScrollArea()
        scroll.setWidget(top)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        bl.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono())
        bl.addWidget(self._log)

        splitter.addWidget(scroll)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    # --- Extent group ---
    def _build_extent_group(self) -> QGroupBox:
        gb = QGroupBox("예측 공간 범위 (Extent)")
        lay = QVBoxLayout(gb)

        # ── Source radio buttons ──────────────────────────────────────────
        src_row = QHBoxLayout()
        self._radio_ext_raster = QRadioButton("래스터 자동 감지")
        self._radio_ext_shp    = QRadioButton("SHP 파일")
        self._radio_ext_manual = QRadioButton("수동 입력")
        self._radio_ext_raster.setChecked(True)
        bg_ext = QButtonGroup(self)
        for rb in (self._radio_ext_raster, self._radio_ext_shp, self._radio_ext_manual):
            bg_ext.addButton(rb)
            src_row.addWidget(rb)
        src_row.addStretch()
        lay.addLayout(src_row)

        # ── Stacked sub-widgets (one per source) ─────────────────────────
        self._ext_stack = QStackedWidget()

        # Page 0 – raster auto  (just an info label + convenience button)
        p0 = QWidget()
        p0_lay = QHBoxLayout(p0)
        p0_lay.setContentsMargins(4, 2, 4, 2)
        p0_lay.addWidget(QLabel("첫 번째 래스터의 Extent를 실행 시 자동으로 사용합니다."))
        btn_load_raster = QPushButton("래스터에서 수동으로 불러오기…")
        btn_load_raster.clicked.connect(self._load_extent_from_raster)
        p0_lay.addWidget(btn_load_raster)
        p0_lay.addStretch()
        self._ext_stack.addWidget(p0)

        # Page 1 – SHP file
        p1 = QWidget()
        p1_lay = QHBoxLayout(p1)
        p1_lay.setContentsMargins(4, 2, 4, 2)
        p1_lay.addWidget(QLabel("SHP 파일:"))
        self._shp_extent_edit = QLineEdit()
        self._shp_extent_edit.setPlaceholderText("SHP / GPKG / GeoJSON …")
        p1_lay.addWidget(self._shp_extent_edit)
        btn_shp = QPushButton("찾기")
        btn_shp.clicked.connect(self._browse_shp_extent)
        p1_lay.addWidget(btn_shp)
        self._ext_stack.addWidget(p1)

        # Page 2 – manual spinboxes
        p2 = QWidget()
        p2_lay = QHBoxLayout(p2)
        p2_lay.setContentsMargins(4, 2, 4, 2)
        for label, attr in [("MinX:", "_ext_minx"), ("MinY:", "_ext_miny"),
                             ("MaxX:", "_ext_maxx"), ("MaxY:", "_ext_maxy")]:
            p2_lay.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.setSingleStep(1000)
            spin.setValue(0.0)
            spin.setFixedWidth(120)
            setattr(self, attr, spin)
            p2_lay.addWidget(spin)
        p2_lay.addStretch()
        self._ext_stack.addWidget(p2)

        # Wire radio → stack page
        self._radio_ext_raster.toggled.connect(lambda on: on and self._ext_stack.setCurrentIndex(0))
        self._radio_ext_shp.toggled.connect(   lambda on: on and self._ext_stack.setCurrentIndex(1))
        self._radio_ext_manual.toggled.connect(lambda on: on and self._ext_stack.setCurrentIndex(2))

        lay.addWidget(self._ext_stack)

        # ── Resolution ───────────────────────────────────────────────────
        row2 = QHBoxLayout()
        self._chk_auto_res = QCheckBox("해상도 자동 감지 (래스터 기준)")
        self._chk_auto_res.setChecked(True)
        self._chk_auto_res.toggled.connect(self._toggle_res_input)
        row2.addWidget(self._chk_auto_res)
        row2.addSpacing(20)
        row2.addWidget(QLabel("수동 해상도 (m):"))
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.1, 100000)
        self._res_spin.setValue(30.0)
        self._res_spin.setDecimals(2)
        self._res_spin.setEnabled(False)
        row2.addWidget(self._res_spin)
        row2.addStretch()
        lay.addLayout(row2)

        return gb

    # --- Raster group ---
    def _build_raster_group(self) -> QGroupBox:
        gb = QGroupBox("독립변수 래스터 (학습과 동일하게 설정)")
        lay = QVBoxLayout(gb)
        self._raster_table = RasterTableWidget()
        self._raster_table.setMinimumHeight(160)
        lay.addWidget(self._raster_table)
        return gb

    # --- Buffer group ---
    def _build_buffer_group(self) -> QGroupBox:
        gb = QGroupBox("버퍼 설정 (학습과 동일하게 설정)")
        lay = QVBoxLayout(gb)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("버퍼 방법:"))
        self._radio_circle = QRadioButton("원형 (Circle)")
        self._radio_moore = QRadioButton("격자 (Moore / BBox)")
        self._radio_circle.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._radio_circle)
        bg.addButton(self._radio_moore)
        row1.addWidget(self._radio_circle)
        row1.addWidget(self._radio_moore)
        row1.addStretch()
        lay.addLayout(row1)

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

        self._buf_preview = QLabel()
        self._buf_min.valueChanged.connect(self._upd_preview)
        self._buf_max.valueChanged.connect(self._upd_preview)
        self._buf_step.valueChanged.connect(self._upd_preview)
        self._upd_preview()
        lay.addWidget(self._buf_preview)

        return gb

    # --- Output group ---
    def _build_output_group(self) -> QGroupBox:
        gb = QGroupBox("출력")
        lay = QVBoxLayout(gb)

        row = QHBoxLayout()
        row.addWidget(QLabel("출력 CSV:"))
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("pred_data.csv")
        row.addWidget(self._output_edit)
        btn = QPushButton("찾기")
        btn.clicked.connect(self._browse_output)
        row.addWidget(btn)
        lay.addLayout(row)

        note = QLabel("※ grid_meta JSON 파일이 CSV와 함께 자동 저장됩니다.")
        note.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(note)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  예측 전처리 실행")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setStyleSheet("font-weight: bold; background: #FF9800; color: white;")
        self._btn_run.clicked.connect(self._run)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        lay.addLayout(btn_row)

        return gb

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _toggle_res_input(self, checked: bool):
        self._res_spin.setEnabled(not checked)

    def _browse_shp_extent(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "벡터 파일 선택", "",
            "벡터 파일 (*.shp *.gpkg *.geojson *.json);;모든 파일 (*.*)"
        )
        if path:
            self._shp_extent_edit.setText(path)

    def _load_extent_from_raster(self):
        """편의 버튼: 래스터를 골라 수동 입력 칸을 채우고 수동 모드로 전환."""
        path, _ = QFileDialog.getOpenFileName(
            self, "래스터 파일 선택", "",
            "래스터 (*.tif *.tiff *.img *.vrt);;모든 파일 (*.*)"
        )
        if not path:
            return
        try:
            info = get_raster_info(path)
            b = info["bounds"]
            self._ext_minx.setValue(b.left)
            self._ext_miny.setValue(b.bottom)
            self._ext_maxx.setValue(b.right)
            self._ext_maxy.setValue(b.top)
            self._res_spin.setValue(info["res_x"])
            self._radio_ext_manual.setChecked(True)
            self._chk_auto_res.setChecked(False)
            self._log.append(f"Extent 로드: {b}")
            self._log.append(f"해상도: {info['res_x']}m")
        except Exception as e:
            QMessageBox.warning(self, "오류", str(e))

    def _upd_preview(self):
        sizes = self._get_buffer_sizes()
        self._buf_preview.setText(f"  → 버퍼 크기 목록: {sizes} m")

    def _get_buffer_sizes(self) -> list[float]:
        bmin = self._buf_min.value()
        bmax = self._buf_max.value()
        bstep = self._buf_step.value()
        return [float(v) for v in range(bmin, bmax + 1, bstep)] if bstep > 0 else [float(bmin)]

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "출력 CSV 저장", "", "CSV (*.csv)")
        if path:
            if not path.endswith(".csv"):
                path += ".csv"
            self._output_edit.setText(path)

    def _run(self):
        configs = self._raster_table.get_configs()
        if not configs:
            QMessageBox.warning(self, "입력 오류", "래스터 파일을 추가하세요.")
            return
        if not self._output_edit.text().strip():
            QMessageBox.warning(self, "입력 오류", "출력 CSV 경로를 입력하세요.")
            return

        # Extent
        if self._radio_ext_raster.isChecked():
            try:
                info = get_raster_info(configs[0]["path"])
                b = info["bounds"]
                extent = (b.left, b.bottom, b.right, b.top)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"Extent 자동 감지 실패: {e}")
                return
        elif self._radio_ext_shp.isChecked():
            shp_path = self._shp_extent_edit.text().strip()
            if not shp_path:
                QMessageBox.warning(self, "입력 오류", "SHP 파일을 선택하세요.")
                return
            try:
                import geopandas as gpd
                gdf = gpd.read_file(shp_path)
                b = gdf.total_bounds  # (minx, miny, maxx, maxy)
                extent = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                self._log.append(
                    f"SHP Extent 로드: MinX={extent[0]:.4f} MinY={extent[1]:.4f} "
                    f"MaxX={extent[2]:.4f} MaxY={extent[3]:.4f}"
                )
            except Exception as e:
                QMessageBox.critical(self, "오류", f"SHP Extent 읽기 실패: {e}")
                return
        else:  # manual
            extent = (
                self._ext_minx.value(),
                self._ext_miny.value(),
                self._ext_maxx.value(),
                self._ext_maxy.value(),
            )

        resolution = None if self._chk_auto_res.isChecked() else self._res_spin.value()

        params = {
            "raster_configs": configs,
            "buffer_sizes": self._get_buffer_sizes(),
            "buffer_method": "circle" if self._radio_circle.isChecked() else "moore",
            "extent": extent,
            "resolution": resolution,
            "output_csv": self._output_edit.text().strip(),
        }

        self._log.clear()
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)

        self._worker = PredPreprocessWorker(params, parent=self)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, csv_path: str, grid_meta: dict):
        self._progress.setValue(100)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[완료] {csv_path}")
        QMessageBox.information(self, "완료", f"예측 전처리 완료!\n{csv_path}")
        self.output_path_changed.emit(csv_path, grid_meta)

    def _on_error(self, msg: str):
        self._progress.setValue(0)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[오류]\n{msg}")
        QMessageBox.critical(self, "오류", msg[:500])


def _mono():
    f = QFont("Consolas")
    f.setPointSize(9)
    return f
