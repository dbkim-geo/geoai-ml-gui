"""
Tab 2 – 모델 학습.

입력: 전처리 CSV + 종속변수 컬럼 + 모델 선택
처리: RandomizedSearchCV 튜닝 → CV 성능 평가 → 모델 저장
출력: 성능 테이블, 차트, (선택) SHAP figure
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.training import get_available_models
from src.gui.widgets.mpl_canvas import MplCanvas
from src.workers.train_worker import TrainWorker


class TrainTab(QWidget):
    """Tab 2 위젯."""

    model_trained = pyqtSignal(dict)  # {model_name: result} → Tab4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainWorker | None = None
        self._results: dict = {}
        self._chart_paths: list[str] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Top: settings ──────────────────────────────────────────────
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.addWidget(self._build_input_group())
        tl.addWidget(self._build_model_group())
        tl.addWidget(self._build_train_options_group())
        tl.addWidget(self._build_output_group())

        scroll = QScrollArea()
        scroll.setWidget(top)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)

        # ── Bottom: results ────────────────────────────────────────────
        bottom = QSplitter(Qt.Horizontal)

        # Left: progress + log + metrics table
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        left_l.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono())
        left_l.addWidget(self._log)

        left_l.addWidget(QLabel("성능 결과 테이블"))
        self._metrics_table = QTableWidget()
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        left_l.addWidget(self._metrics_table)

        # Right: charts
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)

        chart_toolbar = QHBoxLayout()
        chart_toolbar.addWidget(QLabel("차트 선택:"))
        self._chart_combo = QComboBox()
        self._chart_combo.currentIndexChanged.connect(self._show_selected_chart)
        chart_toolbar.addWidget(self._chart_combo)
        chart_toolbar.addStretch()
        right_l.addLayout(chart_toolbar)

        self._canvas = MplCanvas(width=8, height=5)
        right_l.addWidget(self._canvas)

        bottom.addWidget(left_w)
        bottom.addWidget(right_w)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)

        splitter.addWidget(scroll)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter)

    # --- Input group ---
    def _build_input_group(self) -> QGroupBox:
        gb = QGroupBox("입력 데이터")
        lay = QVBoxLayout(gb)

        # CSV
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("학습 CSV:"))
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText("전처리 완료된 CSV 파일")
        self._csv_edit.textChanged.connect(self._on_csv_path_changed)
        row1.addWidget(self._csv_edit)
        btn = QPushButton("찾기")
        btn.clicked.connect(self._browse_csv)
        row1.addWidget(btn)
        lay.addLayout(row1)

        # Target column
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("종속변수 컬럼:"))
        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._target_combo.setMinimumWidth(200)
        self._target_combo.lineEdit().setPlaceholderText("CSV 로드 후 선택")
        row2.addWidget(self._target_combo)
        row2.addSpacing(20)

        row2.addWidget(QLabel("작업 유형:"))
        self._radio_reg = QRadioButton("회귀")
        self._radio_cls = QRadioButton("분류")
        self._radio_reg.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._radio_reg)
        bg.addButton(self._radio_cls)
        self._radio_reg.toggled.connect(self._refresh_model_list)
        row2.addWidget(self._radio_reg)
        row2.addWidget(self._radio_cls)
        row2.addStretch()
        lay.addLayout(row2)

        return gb

    # --- Model selection ---
    def _build_model_group(self) -> QGroupBox:
        gb = QGroupBox("모델 선택")
        outer = QVBoxLayout(gb)

        # Select all / none buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("전체 선택")
        btn_none = QPushButton("전체 해제")
        btn_all.clicked.connect(lambda: self._set_all_checks(True))
        btn_none.clicked.connect(lambda: self._set_all_checks(False))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # Checkbox scroll area
        self._model_check_widget = QWidget()
        self._model_check_layout = QHBoxLayout(self._model_check_widget)
        self._model_check_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._checkboxes: dict[str, QCheckBox] = {}

        scroll = QScrollArea()
        scroll.setWidget(self._model_check_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(90)
        outer.addWidget(scroll)

        self._refresh_model_list()
        return gb

    def _refresh_model_list(self):
        task = "regression" if self._radio_reg.isChecked() else "classification"
        models = get_available_models(task)

        # Clear existing checkboxes
        for cb in self._checkboxes.values():
            self._model_check_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        # Create column layout
        n_cols = 4
        grid_w = QWidget()
        from PyQt5.QtWidgets import QGridLayout
        grid = QGridLayout(grid_w)
        for i, name in enumerate(models):
            cb = QCheckBox(name)
            cb.setChecked(True)
            row, col = divmod(i, n_cols)
            grid.addWidget(cb, row, col)
            self._checkboxes[name] = cb

        # Remove old widget from scroll
        old = self._model_check_layout.itemAt(0)
        if old and old.widget():
            old.widget().deleteLater()
            self._model_check_layout.removeItem(old)

        self._model_check_layout.addWidget(grid_w)

    def _set_all_checks(self, state: bool):
        for cb in self._checkboxes.values():
            cb.setChecked(state)

    # --- Training options ---
    def _build_train_options_group(self) -> QGroupBox:
        gb = QGroupBox("학습 옵션")
        lay = QVBoxLayout(gb)

        row1 = QHBoxLayout()
        self._chk_tune = QCheckBox("하이퍼파라미터 튜닝 (RandomizedSearchCV)")
        self._chk_tune.setChecked(True)
        row1.addWidget(self._chk_tune)

        row1.addSpacing(20)
        row1.addWidget(QLabel("랜덤서치 횟수:"))
        self._n_iter = QSpinBox()
        self._n_iter.setRange(1, 200)
        self._n_iter.setValue(10)
        row1.addWidget(self._n_iter)

        row1.addSpacing(20)
        row1.addWidget(QLabel("CV Fold:"))
        self._cv_folds = QSpinBox()
        self._cv_folds.setRange(2, 20)
        self._cv_folds.setValue(5)
        row1.addWidget(self._cv_folds)

        row1.addSpacing(20)
        row1.addWidget(QLabel("테스트 비율:"))
        self._test_size = QDoubleSpinBox()
        self._test_size.setRange(0.05, 0.5)
        self._test_size.setSingleStep(0.05)
        self._test_size.setValue(0.2)
        self._test_size.setDecimals(2)
        row1.addWidget(self._test_size)
        row1.addStretch()
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._chk_shap = QCheckBox("학습 후 SHAP 분석 수행 (시간 소요)")
        row2.addWidget(self._chk_shap)
        row2.addStretch()
        lay.addLayout(row2)

        return gb

    # --- Output group ---
    def _build_output_group(self) -> QGroupBox:
        gb = QGroupBox("출력")
        lay = QVBoxLayout(gb)

        row = QHBoxLayout()
        row.addWidget(QLabel("출력 디렉토리:"))
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("모델/차트/SHAP 저장 폴더")
        row.addWidget(self._out_dir_edit)
        btn = QPushButton("찾기")
        btn.clicked.connect(self._browse_output_dir)
        row.addWidget(btn)
        lay.addLayout(row)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  학습 실행")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setStyleSheet("font-weight: bold; background: #4CAF50; color: white;")
        self._btn_run.clicked.connect(self._run)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_run)
        lay.addLayout(btn_row)

        return gb

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "학습 CSV 선택", "", "CSV (*.csv)")
        if path:
            self._csv_edit.setText(path)
            # textChanged 시그널이 _on_csv_path_changed 호출

    def _on_csv_path_changed(self, path: str):
        """CSV 파일이 바뀔 때 컬럼 목록을 드롭다운에 채운다."""
        path = path.strip()
        if not path or not path.lower().endswith(".csv"):
            return
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=0)  # 헤더만 읽기
            cols = list(df.columns)
            # 일반적으로 fid, grid_x/y 같은 좌표·식별자 컬럼 제외 제안
            exclude = {"fid", "FID", "grid_x", "grid_y"}
            candidate_cols = [c for c in cols if c not in exclude]

            prev = self._target_combo.currentText()
            self._target_combo.clear()
            self._target_combo.addItems(candidate_cols)
            if prev in candidate_cols:
                self._target_combo.setCurrentText(prev)
            self._log.append(f"컬럼 로드: {candidate_cols}")
        except Exception as e:
            self._log.append(f"[경고] CSV 컬럼 읽기 실패: {e}")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "출력 디렉토리 선택")
        if path:
            self._out_dir_edit.setText(path)

    def _run(self):
        csv = self._csv_edit.text().strip()
        target = self._target_combo.currentText().strip()
        out_dir = self._out_dir_edit.text().strip()

        if not csv:
            QMessageBox.warning(self, "입력 오류", "학습 CSV 파일을 선택하세요.")
            return
        if not target:
            QMessageBox.warning(self, "입력 오류", "종속변수 컬럼명을 입력하세요.")
            return
        if not out_dir:
            QMessageBox.warning(self, "입력 오류", "출력 디렉토리를 선택하세요.")
            return

        selected = [n for n, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "입력 오류", "모델을 하나 이상 선택하세요.")
            return

        params = {
            "csv_path": csv,
            "target_col": target,
            "selected_models": selected,
            "task_type": "regression" if self._radio_reg.isChecked() else "classification",
            "test_size": self._test_size.value(),
            "cv_folds": self._cv_folds.value(),
            "tune": self._chk_tune.isChecked(),
            "n_iter": self._n_iter.value(),
            "output_dir": out_dir,
            "run_shap": self._chk_shap.isChecked(),
        }

        self._log.clear()
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)
        self._chart_combo.clear()
        self._canvas.clear()
        self._metrics_table.setRowCount(0)

        self._worker = TrainWorker(params, parent=self)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, results: dict):
        self._progress.setValue(100)
        self._btn_run.setEnabled(True)
        self._results = results
        self._log.append("\n[완료] 학습 종료")
        self._populate_metrics_table(results)
        self._load_charts(results)
        self.model_trained.emit(results)
        QMessageBox.information(self, "완료", "모델 학습 완료!")

    def _on_error(self, msg: str):
        self._progress.setValue(0)
        self._btn_run.setEnabled(True)
        self._log.append(f"\n[오류]\n{msg}")
        QMessageBox.critical(self, "오류", msg[:500])

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------
    def _populate_metrics_table(self, results: dict):
        if not results:
            return

        # Collect all metric keys
        all_keys = []
        for res in results.values():
            for k in res["metrics"]:
                if k not in all_keys:
                    all_keys.append(k)

        self._metrics_table.setColumnCount(1 + len(all_keys))
        self._metrics_table.setHorizontalHeaderLabels(["Model"] + all_keys)
        self._metrics_table.setRowCount(len(results))

        for row, (mname, res) in enumerate(results.items()):
            self._metrics_table.setItem(row, 0, QTableWidgetItem(mname))
            for col, key in enumerate(all_keys, start=1):
                val = res["metrics"].get(key)
                txt = f"{val:.4f}" if val is not None else "-"
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self._metrics_table.setItem(row, col, item)

        self._metrics_table.resizeColumnsToContents()

    def _load_charts(self, results: dict):
        out_dir = self._out_dir_edit.text().strip()
        chart_dir = os.path.join(out_dir, "charts")
        if not os.path.isdir(chart_dir):
            return

        self._chart_paths = []
        chart_names = []

        # performance comparison
        p = os.path.join(chart_dir, "performance_comparison.png")
        if os.path.isfile(p):
            self._chart_paths.append(p)
            chart_names.append("성능 비교")

        # actual vs predicted
        p2 = os.path.join(chart_dir, "actual_vs_predicted.png")
        if os.path.isfile(p2):
            self._chart_paths.append(p2)
            chart_names.append("실측 vs 예측")

        # feature importance per model
        for mname in results:
            safe = mname.replace(" ", "_")
            p3 = os.path.join(chart_dir, f"{safe}_feature_importance.png")
            if os.path.isfile(p3):
                self._chart_paths.append(p3)
                chart_names.append(f"FI: {mname}")

        self._chart_combo.addItems(chart_names)
        if chart_names:
            self._show_selected_chart(0)

    def _show_selected_chart(self, index: int):
        if 0 <= index < len(self._chart_paths):
            self._canvas.show_image(self._chart_paths[index])

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------
    def set_input_csv(self, path: str):
        self._csv_edit.setText(path)


def _mono():
    f = QFont("Consolas")
    f.setPointSize(9)
    return f
