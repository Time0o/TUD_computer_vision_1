#!/usr/bin/env python3

import argparse
import sys

import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def mat_to_pixmap(mat: np.ndarray) -> QtGui.QPixmap:
    assert len(mat.shape) == 2

    image = QtGui.QImage(
        mat.data, mat.shape[1], mat.shape[0], mat.strides[0],
        QtGui.QImage.Format_Grayscale8)

    pixmap = QtGui.QPixmap()
    pixmap.convertFromImage(image)

    return pixmap


class PolynomialSelectionWidget(QtWidgets.QGraphicsView):
    SIZE = QtCore.QSize(512, 512)

    MINIMUM_SELECTOR_COLOR = QtGui.QColor(0, 0, 255)
    MAXIMUM_SELECTOR_COLOR = QtGui.QColor(255, 0, 0)
    POLYNOMIAL_LINEWIDTH = 3
    POLYNOMIAL_COLOR = QtGui.QColor(0, 255, 0)

    INITIAL_MINIMUM_SELECTOR_POS = QtCore.QPoint(SIZE.width() // 4,
                                                 SIZE.height() // 2)

    INITIAL_MAXIMUM_SELECTOR_POS = QtCore.QPoint(3 * (SIZE.width() // 4),
                                                 SIZE.height() // 2)

    minimumChanged = QtCore.pyqtSignal(QtCore.QPoint)
    maximumChanged = QtCore.pyqtSignal(QtCore.QPoint)

    class ExtremumSelector(QtWidgets.QPushButton):
        RADIUS = 10
        STYLESHEET = 'QPushButton {{border: 2px; background: {}}}'

        extremumChanged = QtCore.pyqtSignal()

        def __init__(self, color: QtGui.QColor, parent=None):
            super().__init__(parent)

            self._color = color

            self.setFixedWidth(2 * self.RADIUS)
            self.setFixedHeight(2 * self.RADIUS)

            rect = QtCore.QRect(0, 0, 2 * self.RADIUS, 2 * self.RADIUS)
            self.setMask(QtGui.QRegion(rect, QtGui.QRegion.Ellipse))

            self._lastMouseEventPos = None

        def paintEvent(self, event: QtGui.QPaintEvent):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            pen = QtGui.QPen(QtCore.Qt.black)

            brush = QtGui.QBrush(self._color)

            painter.setPen(pen)
            painter.setBrush(brush)

            margins = QtCore.QMargins(2, 2, 2, 2)
            painter.drawEllipse(self.rect().marginsRemoved(margins))

            painter.end()

        def mousePressEvent(self, event: QtGui.QMouseEvent):
            self._lastMouseEventPos = event.globalPos()

            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
            mouseEventPos = event.globalPos()
            delta = mouseEventPos - self._lastMouseEventPos

            currentPos = self.mapToGlobal(self.pos())
            newPos = self.mapFromGlobal(currentPos + delta)

            if self.parent().sceneRect().contains(newPos):
                self.move(newPos)
            else:
                self.releaseMouse()
                return

            self._lastMouseEventPos = mouseEventPos

            self.extremumChanged.emit() # TODO

            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
            self._lastMouseEventPos = None

            super().mouseReleaseEvent(event)

        def extremum(self) -> QtCore.QPoint:
            tlPos = self.pos()

            return QtCore.QPoint(tlPos.x() + self.RADIUS,
                                 tlPos.y() + self.RADIUS)

    def __init__(self):
        super().__init__()

        self.setFixedSize(self.SIZE)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        # disable scrollbars
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # construct scene
        self._scene = QtWidgets.QGraphicsScene()
        self._scene.setSceneRect(QtCore.QRectF(QtCore.QPointF(0., 0.),
                                               QtCore.QSizeF(self.SIZE)))

        # polynomial graphics item
        self._polynomialGraphicsPathItem = QtWidgets.QGraphicsPathItem()

        pen = QtGui.QPen(self.POLYNOMIAL_COLOR)
        pen.setWidth(self.POLYNOMIAL_LINEWIDTH)
        self._polynomialGraphicsPathItem.setPen(pen)

        self._scene.addItem(self._polynomialGraphicsPathItem)

        # "coordinate axes"
        self._scene.addLine(0, self.SIZE.height() // 2,
                            self.SIZE.width(), self.SIZE.height() // 2)

        self._scene.addLine(self.SIZE.width() // 2, 0,
                            self.SIZE.width() // 2, self.SIZE.height())

        # minimum and maximum selection buttons
        self._minimumSelector = self.ExtremumSelector(
            self.MINIMUM_SELECTOR_COLOR, parent=self)

        self._maximumSelector = self.ExtremumSelector(
            self.MAXIMUM_SELECTOR_COLOR, parent=self)

        self._minimumSelector.extremumChanged.connect(self._updatePolynomial)
        self._maximumSelector.extremumChanged.connect(self._updatePolynomial)

        self._minimumSelector.move(
            self.INITIAL_MINIMUM_SELECTOR_POS.x(),
            self.INITIAL_MINIMUM_SELECTOR_POS.y() - self.ExtremumSelector.RADIUS)

        self._maximumSelector.move(
            self.INITIAL_MAXIMUM_SELECTOR_POS.x(),
            self.INITIAL_MAXIMUM_SELECTOR_POS.y() - self.ExtremumSelector.RADIUS)

        self.setScene(self._scene)

    @QtCore.pyqtSlot()
    def _updatePolynomial(self):
        minX = float(self._minimumSelector.extremum().x() //
                     (self.SIZE.width() // 256))

        minY = float(self._minimumSelector.extremum().y() //
                     (self.SIZE.height() // 256))

        maxX = float(self._maximumSelector.extremum().x() //
                     (self.SIZE.width() // 256))

        maxY = float(self._maximumSelector.extremum().y() //
                     (self.SIZE.height() // 256))

        dx = maxX - minX
        if dx == 0.:
            dx = 0.001

        dy = maxY - minY

        x0 = (minX + maxX) / 2

        a = -2. * dy / dx**3
        b = -3. * dy / (2. * dx)
        c = (minY + maxY) / 2. + b * x0

        p = lambda x: a * (x - x0)**3 - b * x + c

        scaleX, scaleY = self.SIZE.width() // 256, self.SIZE.height() // 256

        firstReached = False
        lastReached = False

        for x in range(0, 256):
            xScaled = x * scaleX
            yScaled = p(x) * scaleY

            if not firstReached:
                if yScaled < self.SIZE.height():
                    firstPoint = QtCore.QPointF(xScaled, self.SIZE.height())
                    path = QtGui.QPainterPath(firstPoint)
                    firstReached = True
                else:
                    continue

            if yScaled >= self.SIZE.height():
                if not lastReached:
                    path.lineTo(QtCore.QPointF(xScaled, self.SIZE.height()))
                    lastReached = True
            else:
                path.lineTo(QtCore.QPointF(xScaled, yScaled))

        self._polynomialGraphicsPathItem.setPath(path)
        self._scene.update(self._scene.itemsBoundingRect())


class SolarizationMainWindow(QtWidgets.QMainWindow):
    GRAPHICSVIEW_SIZE = QtCore.QSize(512, 512)

    def __init__(self, mat: np.ndarray):
        super().__init__()

        # initialize image pixmap
        self._graphicsPixmapItem = QtWidgets.QGraphicsPixmapItem()

        self._updateMat(mat)

        # construct scene
        self._graphicsScene = QtWidgets.QGraphicsScene()
        self._graphicsScene.addItem(self._graphicsPixmapItem)

        self._graphicsView = QtWidgets.QGraphicsView()
        self._graphicsView.setScene(self._graphicsScene)

        self._graphicsView.setMinimumSize(self.GRAPHICSVIEW_SIZE)

        # construct polynomial selection widget
        self._polynomialSelectionWidget = PolynomialSelectionWidget()

        # construct central widget
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._graphicsView)

        rightLayout = QtWidgets.QVBoxLayout()
        rightLayout.addWidget(self._polynomialSelectionWidget)
        rightLayout.addStretch()

        layout.addItem(rightLayout)

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.resize(self.size())

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self._graphicsView.fitInView(self._graphicsScene.sceneRect(),
                                     QtCore.Qt.KeepAspectRatio)

    @QtCore.pyqtSlot(np.ndarray)
    def _updateMat(self, mat: np.ndarray):
        self._mat = mat

        pixmap = mat_to_pixmap(self._mat)
        self._graphicsPixmapItem.setPixmap(pixmap)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('image', metavar='IMAGE',
                        help="input image")

    parser.add_argument('-i', '--interactive', action='store_true',
                        help="enter interactive mode")

    parser.add_argument('--low', type=int,
                        help="polynomial low point")

    parser.add_argument('--high', type=int,
                        help="polynomial high point")

    args = parser.parse_args()

    if not args.interactive:
        if args.low is None or args.high is None:
            print("Either --interactive or --low and --high must be specified",
                  file=sys.stderr)
            sys.exit(1)
    else:
        if not args.low is None and args.high is None:
            print("--interactive and --low/--high may not be used together",
                  file=sys.stderr)
            sys.exit(1)

    # read input image
    mat = cv.imread(args.image, cv.IMREAD_GRAYSCALE)

    if mat is None:
        print("Failed to read input image",
              file=sys.stderr)
        sys.exit(1)

    # start interactive mode
    if args.interactive:
        app = QtWidgets.QApplication(sys.argv)

        def excepthook(etype, eval, etraceback):
            print("{}: {}".format(etype.__name__, eval), file=sys.stderr)

            app.closeAllWindows()

        sys.excepthook = excepthook

        mainWindow = SolarizationMainWindow(mat)
        mainWindow.show()

        sys.exit(app.exec())
    else:
        pass # TODO
