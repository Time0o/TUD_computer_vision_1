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


class SolarizationMainWindow(QtWidgets.QMainWindow):
    def __init__(self, mat: np.ndarray):
        super().__init__()

        self._graphicsPixmapItem = QtWidgets.QGraphicsPixmapItem()

        self.update_mat(mat)

        self._graphicsScene = QtWidgets.QGraphicsScene()
        self._graphicsScene.addItem(self._graphicsPixmapItem)

        self._graphicsView = QtWidgets.QGraphicsView()
        self._graphicsView.setScene(self._graphicsScene)

        self.setCentralWidget(self._graphicsView)

    @QtCore.pyqtSlot(np.ndarray)
    def update_mat(self, mat: np.ndarray):
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

        mainWindow = SolarizationMainWindow(mat)
        mainWindow.show()

        sys.exit(app.exec())
    else:
        pass # TODO
