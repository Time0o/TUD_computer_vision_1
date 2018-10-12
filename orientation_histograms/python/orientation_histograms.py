#!/usr/bin/env python3

import sys
import traceback

import cv2 as cv
import numpy as np
import qdarkstyle
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


class ImageDisplayWidget(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        # main image
        self._mainGraphicsPixmapItem = QtWidgets.QGraphicsPixmapItem()

        self._mainScene = QtWidgets.QGraphicsScene()
        self._mainScene.addItem(self._mainGraphicsPixmapItem)

        self._mainView = QtWidgets.QGraphicsView()
        self._mainView.setScene(self._mainScene)

        # histogram magnitude
        self._gradientMagnitudeGraphicsPixmapItem = QtWidgets.QGraphicsPixmapItem()

        self._gradientMagnitudeScene = QtWidgets.QGraphicsScene()
        self._gradientMagnitudeScene.addItem(
            self._gradientMagnitudeGraphicsPixmapItem)

        self._gradientMagnitudeView = QtWidgets.QGraphicsView()
        self._gradientMagnitudeView.setScene(self._gradientMagnitudeScene)

        # histogram phase
        self._gradientPhaseGraphicsPixmapItem = QtWidgets.QGraphicsPixmapItem()

        self._gradientPhaseScene = QtWidgets.QGraphicsScene()
        self._gradientPhaseScene.addItem(self._gradientPhaseGraphicsPixmapItem)

        self._gradientPhaseView = QtWidgets.QGraphicsView()
        self._gradientPhaseView.setScene(self._gradientPhaseScene)

        # combined gradient view widget
        gradientWidget = QtWidgets.QWidget()
        gradientWidgetLayout = QtWidgets.QVBoxLayout()
        gradientWidgetLayout.addWidget(QtWidgets.QLabel('Gradient Magnitude'))
        gradientWidgetLayout.addWidget(self._gradientMagnitudeView)
        gradientWidgetLayout.addWidget(QtWidgets.QLabel('Gradient Phase'))
        gradientWidgetLayout.addWidget(self._gradientPhaseView)
        gradientWidget.setLayout(gradientWidgetLayout)

        # layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._mainView, 2)
        layout.addWidget(gradientWidget, 1)
        self.setLayout(layout)

    def updateImage(self, mat: np.ndarray):
        pixmap = matToPixmap(matToGrayscale(mat))
        self._mainGraphicsPixmapItem.setPixmap(pixmap)

    def updateGradientMagnitudeImage(self, mat: np.ndarray):
        pixmap = matToPixmap(matToGrayscale(mat))
        self._gradientMagnitudeGraphicsPixmapItem.setPixmap(pixmap)

    def updateGradientPhaseImage(self, mat: np.ndarray):
        pixmap = matToPixmap(matToGrayscale(mat))
        self._gradientPhaseGraphicsPixmapItem.setPixmap(pixmap)

    def fitImages(self):
        self._mainView.fitInView(
            self._mainScene.itemsBoundingRect(),
            QtCore.Qt.KeepAspectRatio)

        self._gradientPhaseView.fitInView(
            self._gradientPhaseScene.itemsBoundingRect(),
            QtCore.Qt.KeepAspectRatio)

        self._gradientMagnitudeView.fitInView(
            self._gradientMagnitudeScene.itemsBoundingRect(),
            QtCore.Qt.KeepAspectRatio)


class OrientationHistogramsMainWindow(QtWidgets.QMainWindow):
    def __init__(self, mat: np.ndarray):
        super().__init__()

        # display image and gradient magnitude/phase
        self._imageDisplayWidget = ImageDisplayWidget()
        self.setCentralWidget(self._imageDisplayWidget)

        self._imageDisplayWidget.updateImage(mat)

        magnitude, phase = matGradient(mat)
        self._imageDisplayWidget.updateGradientMagnitudeImage(magnitude)
        self._imageDisplayWidget.updateGradientPhaseImage(phase)

    def initializeDisplay(self):
        self._imageDisplayWidget.fitImages()


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
