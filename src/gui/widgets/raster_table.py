"""
래스터 파일 목록 테이블 위젯.
각 행: 파일경로 | 이름 | 유형(연속형/범주형) | CRS
"""
from __future__ import annotations

import os

import rasterio
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RasterTableWidget(QWidget):
    """
    래스터 파일 추가/삭제 및 연속형/범주형 선택 가능한 테이블 위젯.
    """

    TYPE_CONTINUOUS = "연속형 (Continuous)"
    TYPE_CATEGORICAL = "범주형 (Categorical)"

    COLUMNS = ["파일 경로", "이름", "유형", "CRS"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_add = QPushButton("+ 래스터 추가")
        self._btn_remove = QPushButton("- 선택 삭제")
        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove)
        toolbar.addWidget(self._btn_add)
        toolbar.addWidget(self._btn_remove)
        toolbar.addStretch()

        # Table
        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.DoubleClicked)
        self._table.verticalHeader().setVisible(False)

        main_layout.addLayout(toolbar)
        main_layout.addWidget(self._table)

    # ------------------------------------------------------------------
    def _on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "래스터 파일 선택",
            "",
            "래스터 파일 (*.tif *.tiff *.img *.vrt *.nc);;모든 파일 (*.*)",
        )
        for path in paths:
            self._add_row(path)

    def _on_remove(self):
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._table.removeRow(row)

    @staticmethod
    def _read_crs(path: str) -> str:
        """래스터 CRS를 EPSG 코드(또는 WKT 축약) 문자열로 반환."""
        try:
            with rasterio.open(path) as src:
                crs = src.crs
                if crs is None:
                    return "CRS 없음"
                epsg = crs.to_epsg()
                return f"EPSG:{epsg}" if epsg else crs.to_string()[:30]
        except Exception:
            return "읽기 실패"

    def _add_row(self, path: str):
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Path (read-only)
        path_item = QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 0, path_item)

        # Name (editable, default = stem)
        stem = os.path.splitext(os.path.basename(path))[0]
        name_item = QTableWidgetItem(stem)
        self._table.setItem(row, 1, name_item)

        # Type combo
        combo = QComboBox()
        combo.addItems([self.TYPE_CONTINUOUS, self.TYPE_CATEGORICAL])
        self._table.setCellWidget(row, 2, combo)

        # CRS (read-only)
        crs_str = self._read_crs(path)
        crs_item = QTableWidgetItem(crs_str)
        crs_item.setFlags(crs_item.flags() & ~Qt.ItemIsEditable)
        crs_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 3, crs_item)

    # ------------------------------------------------------------------
    def get_configs(self) -> list[dict]:
        """현재 테이블의 래스터 설정 목록 반환."""
        configs = []
        for row in range(self._table.rowCount()):
            path_item = self._table.item(row, 0)
            name_item = self._table.item(row, 1)
            combo: QComboBox = self._table.cellWidget(row, 2)

            if path_item is None:
                continue

            path = path_item.text().strip()
            name = name_item.text().strip() if name_item else os.path.basename(path)
            rtype = (
                "categorical"
                if combo and combo.currentText() == self.TYPE_CATEGORICAL
                else "continuous"
            )
            configs.append({"path": path, "name": name, "type": rtype})
        return configs

    def set_configs(self, configs: list[dict]):
        """외부에서 설정 목록을 로드."""
        self._table.setRowCount(0)
        for cfg in configs:
            self._add_row(cfg.get("path", ""))
            row = self._table.rowCount() - 1
            if "name" in cfg:
                self._table.item(row, 1).setText(cfg["name"])
            combo: QComboBox = self._table.cellWidget(row, 2)
            if cfg.get("type") == "categorical":
                combo.setCurrentText(self.TYPE_CATEGORICAL)

    def row_count(self) -> int:
        return self._table.rowCount()
