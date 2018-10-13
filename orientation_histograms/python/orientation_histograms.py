#!/usr/bin/env python3

import sys
import traceback

import cv2 as cv
import numpy as np
import qdarkstyle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets


def matToGrayscale(mat: np.ndarray) -> np.ndarray:
    if mat.dtype == np.uint8:
        if len(mat.shape) == 2:
            return mat

        if len(mat.shape) == 3:
            if mat.shape[2] == 1:
                return mat

            if mat.shape[2] == 3:
                return cv.cvtColor(mat, cv.COLOR_BGR2GRAY)

    elif mat.dtype == np.float64:
        if len(mat.shape) == 2 or len(mat.shape) == 3 and mat.shape[2] == 1:
            matNormalized = mat / (np.max(mat) / np.float64(255))
            return matNormalized.astype(np.uint8)

    raise ValueError("unsupported image type for grayscale conversion")


def matGradient(mat: np.ndarray):
    matGrayscale = matToGrayscale(mat)

    dx = cv.Sobel(matGrayscale, cv.CV_64F, 1, 0)
    dy = cv.Sobel(matGrayscale, cv.CV_64F, 0, 1)

    magnitude = cv.magnitude(dx, dy)
    phase = cv.phase(dx, dy)

    return magnitude, phase


def matToPixmap(mat: np.ndarray) -> QtGui.QPixmap:
    assert len(mat.shape) == 2

    image = QtGui.QImage(
        mat.data, mat.shape[1], mat.shape[0], mat.strides[0],
        QtGui.QImage.Format_Grayscale8)

    pixmap = QtGui.QPixmap()
    pixmap.convertFromImage(image)

    return pixmap


class TitledViewWidget(QtWidgets.QWidget):
    def __init__(self, title: str, mat: np.ndarray, parent=None):
        super().__init__(parent)

        # display title
        label = QtWidgets.QLabel(title)

        # display image
        pixmap = matToPixmap(matToGrayscale(mat))
        self._graphicsPixmapItem = QtWidgets.QGraphicsPixmapItem(pixmap)

        self._scene = QtWidgets.QGraphicsScene()
        self._scene.addItem(self._graphicsPixmapItem)

        self._view = QtWidgets.QGraphicsView()
        self._view.setScene(self._scene)

        # lay out widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self._view)

        self.setLayout(layout)

    def updateImage(self, mat: np.ndarray):
        pixmap = matToPixmap(matToGrayscale(mat))
        self._graphicsPixmapItem.setPixmap(pixmap)

    def fitImage(self):
        self._view.fitInView(self._scene.itemsBoundingRect(),
                             QtCore.Qt.KeepAspectRatio)


class GradientHistogramWidget(FigureCanvasQTAgg):
    DEFAULT_BINS = 20

    XTICKS = [
        (0, '0'),
        (np.pi / 4, r'$\frac{\pi}{4}$'),
        (np.pi / 2, r'$\frac{\pi}{2}$'),
        (3 * np.pi / 4, r'$\frac{3 \pi}{4}$'),
        (np.pi, '$\pi$')
    ]

    BACKGROUND_COLOR = '#31363b'
    FOREGROUND_COLOR = '#31363b'
    AXES_COLOR = '#76797c'
    PLOT_COLOR = '#308cc6'
    TEXT_COLOR = '#eff0f1'

    def __init__(self,
                 gradientMagnitude: np.ndarray,
                 gradientPhase: np.ndarray,
                 bins: int = None):

        # create figure
        self._fig = Figure()
        self._fig.subplots_adjust(left=.2, bottom=.2)

        # create and label axes
        self._axes = self._fig.add_subplot(111)
        self._axes.set_title('Gradient Phase Histogram', color=self.TEXT_COLOR)
        self._axes.set_xlabel("$\phi$", color=self.TEXT_COLOR)
        self._axes.set_ylabel("$H(\phi)$ (normalized)", color=self.TEXT_COLOR)

        # adjust axes dimension and ticks
        self._axes.set_xlim([0, np.pi])

        xticks, xticklabels = zip(*self.XTICKS)
        self._axes.set_xticks(xticks)
        self._axes.set_xticklabels(xticklabels)

        super().__init__(self._fig)

        # color axes and figure
        self._axes.set_facecolor(self.FOREGROUND_COLOR)

        for spine in self._axes.spines.values():
            spine.set_color(self.AXES_COLOR)

        self._axes.tick_params(axis='x', colors=self.TEXT_COLOR)
        self._axes.tick_params(axis='y', colors=self.TEXT_COLOR)

        self._fig.set_facecolor(self.BACKGROUND_COLOR)

        # generate histogram
        self._gradientMagnitude = gradientMagnitude
        self._gradientPhase = gradientPhase
        self._bins = bins if bins is not None else self.DEFAULT_BINS

        self._updateHistogram()

    def setBins(self, bins: int):
        self._bins = bins

        self._updateHistogram()

    def setGradient(self,
                    gradientMagnitude: np.ndarray,
                    gradientPhase: np.ndarray):

        self._gradientMagnitude = gradientMagnitude
        self._gradientPhase = gradientPhase

        self._updateHistogram()

    def _updateHistogram(self):
        angles = np.linspace(0, np.pi, self._bins)
        sums = np.empty_like(angles)

        for i, angle in enumerate(angles):
            angleDeltas = np.cos(self._gradientPhase - angle)
            sums[i] = np.sum(np.abs(angleDeltas * self._gradientMagnitude))

        sums /= sums.sum()

        self._axes.bar(angles, sums, align='edge')


class OrientationHistogramsMainWindow(QtWidgets.QMainWindow):
    IMAGE_DISPLAY_WIDGET_STRETCH = 1.5
    GRADIENT_WIDGET_STRETCH = 1.

    def __init__(self, mat: np.ndarray):
        super().__init__()

        # calculate image gradient magnitude/phase
        magnitude, phase = matGradient(mat)

        # display image and gradient magnitude/phase
        self._imageDisplayWidget = TitledViewWidget(
            "Input Image", mat)

        self._gradientMagnitudeDisplayWidget = TitledViewWidget(
            "Gradient Magnitude", magnitude)

        self._gradientPhaseDisplayWidget = TitledViewWidget(
            "Gradient Phase", phase)

        # display gradient histogram
        self._gradientHistogramWidget = GradientHistogramWidget(
            magnitude, phase)

        # lay out widgets
        gradientImagesWidget = QtWidgets.QWidget()
        gradientImagesWidgetLayout = QtWidgets.QHBoxLayout()
        gradientImagesWidgetLayout.addWidget(self._gradientMagnitudeDisplayWidget)
        gradientImagesWidgetLayout.addWidget(self._gradientPhaseDisplayWidget)
        gradientImagesWidget.setLayout(gradientImagesWidgetLayout)

        gradientWidget = QtWidgets.QWidget()
        gradientWidgetLayout = QtWidgets.QVBoxLayout()
        gradientWidgetLayout.addWidget(gradientImagesWidget)
        gradientWidgetLayout.addWidget(self._gradientHistogramWidget)
        gradientWidget.setLayout(gradientWidgetLayout)

        centralWidget = QtWidgets.QWidget()
        centralWidgetLayout = QtWidgets.QHBoxLayout()

        centralWidgetLayout.addWidget(self._imageDisplayWidget,
                                      self.IMAGE_DISPLAY_WIDGET_STRETCH)

        centralWidgetLayout.addWidget(gradientWidget,
                                      self.GRADIENT_WIDGET_STRETCH)

        centralWidget.setLayout(centralWidgetLayout)

        self.setCentralWidget(centralWidget)

    def initializeDisplay(self):
        self._imageDisplayWidget.fitImage()
        self._gradientMagnitudeDisplayWidget.fitImage()
        self._gradientPhaseDisplayWidget.fitImage()


if __name__ == '__main__':
    mat = cv.imread(sys.argv[1])

    app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def excepthook(etype, eval, etraceback):
        traceback.print_exception(etype, eval, etraceback)
        app.closeAllWindows()

    sys.excepthook = excepthook

    mainWindow = OrientationHistogramsMainWindow(mat)
    mainWindow.show()

    mainWindow.initializeDisplay()

    sys.exit(app.exec())
