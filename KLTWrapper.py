import numpy as np
import cv2
import itertools

class KLTWrapper:
    def __init__(self):
        self.win_size = 10
        self.status = 0
        self.count = 0
        self.flags = 0

        self.image = None
        self.imgPrevGray = None
        self.H = None

        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24
        self.MAX_COUNT = 0
        self.points0 = None
        self.points1 = None


    def init(self, imgGray):

        (nj, ni) = imgGray.shape

        self.MAX_COUNT = (float(ni) / self.GRID_SIZE_W + 1.0) * (float(nj) / self.GRID_SIZE_H + 1.0)
        self.lk_params = dict(winSize=(self.win_size, self.win_size),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_MAX_ITER| cv2.TERM_CRITERIA_EPS, 20, 0.03))
        self.H = np.identity(3)


    def InitFeatures(self, imgGray):

        self.quality = 0.01
        self.min_distance = 10

        (nj, ni) = imgGray.shape

        self.count = ni / self.GRID_SIZE_W * nj / self.GRID_SIZE_H

        lenI = ni / self.GRID_SIZE_W - 1
        lenJ = nj / self.GRID_SIZE_H - 1
        J = np.arange(lenI*lenJ) / lenJ * self.GRID_SIZE_W + self.GRID_SIZE_W / 2
        I = np.arange(lenJ*lenI) % lenJ * self.GRID_SIZE_H + self.GRID_SIZE_H / 2
        self.points1 = np.expand_dims(zip(J, I), 1).astype(np.float32)
        self.points0, self.points1 = self.points1, self.points0

    def RunTrack(self, image, imgPrev):

        if self.count > 0:
            self.points1, _st, _err = cv2.calcOpticalFlowPyrLK(imgPrev, image, self.points0, None, **self.lk_params)
            good1 = self.points1[_st == 1]
            good2 = self.points0[_st == 1]
            self.count = len(good1)

        if self.count > 10:
            self.makeHomoGraphy(good1, good2)
        else:
            self.H = np.identity(3)
        self.InitFeatures(image)

    def makeHomoGraphy(self, p1, p2):
        self.H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 1.0)
