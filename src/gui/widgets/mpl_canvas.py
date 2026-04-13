"""
Matplotlib figure를 Qt 위젯 안에 임베드하는 캔버스.
"""
from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


class MplCanvas(QWidget):
    """
    Matplotlib Figure를 품고 있는 Qt 위젯.
    이미지 파일 경로를 받아 표시하거나,
    Figure 객체를 직접 전달해 그릴 수 있다.
    """

    def __init__(self, parent=None, width: int = 8, height: int = 5, dpi: int = 100):
        super().__init__(parent)

        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def show_image(self, path: str):
        """PNG/JPEG 이미지를 matplotlib imshow로 표시."""
        import matplotlib.image as mpimg

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        self.canvas.draw()

    def clear(self):
        self.fig.clear()
        self.canvas.draw()
