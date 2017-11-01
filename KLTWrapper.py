import numpy as np
import cv2


class KLTWrapper:
    def __init__(self):
        self.win_size = 10
        self.status = 0;
        self.count = 0;
        self.flags = 0;


        self.eig = None
        self.temp = None
        self.maskimg = None
        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24


    def init(self, imgGray):

        (nj, ni, d) = imgGray.shape

        self.MAX_COUNT = (float(ni) / float(self.GRID_SIZE_W) + 1.0) * (float(nj) / float(self.GRID_SIZE_H) + 1.0)
        self.points0 = (None,) * self.MAX_COUNT
        self.points1 = (None,) * self.MAX_COUNT

    def InitFeatures(self, imgGray):

        quality = 0.01
        min_distance = 10

        (nj, ni, d) = imgGray.shape

        count = ni / self.GRID_SIZE_W * nj / self.GRID_SIZE_H


        cnt = 0
        points1
        for (int i = 0; i < ni / GRID_SIZE_W - 1; ++i) {
        for (int j = 0; j < nj / GRID_SIZE_H - 1; ++j) {
        points[1][cnt].x = i * GRID_SIZE_W + GRID_SIZE_W / 2;
        points[1][cnt++].y = j * GRID_SIZE_H + GRID_SIZE_H / 2;
        }
        }