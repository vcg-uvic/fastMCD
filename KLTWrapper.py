import numpy as np
import cv2
import itertools

class KLTWrapper:
    def __init__(self):
        self.win_size = 10
        self.status = 0;
        self.count = 0;
        self.flags = 0;

        self.image = None
        self.imgPrevGray = None
        self.pyramid = None
        self.prev_pyramid = None
        self.swap_temp = None



        self.eig = None
        self.temp = None
        self.maskimg = None
        self.GRID_SIZE_W = 32
        self.GRID_SIZE_H = 24
        self.MAX_COUNT = 0
        self.points0 = None
        self.points1 = None


    def init(self, imgGray):

        (nj, ni, d) = imgGray.shape

        self.MAX_COUNT = (float(ni) / float(self.GRID_SIZE_W) + 1.0) * (float(nj) / float(self.GRID_SIZE_H) + 1.0)


    def InitFeatures(self, imgGray):

        self.quality = 0.01
        self.min_distance = 10

        (nj, ni, d) = imgGray.shape

        self.count = ni / self.GRID_SIZE_W * nj / self.GRID_SIZE_H



        lenI = ni / self.GRID_SIZE_W - 1
        lenJ = nj / self.GRID_SIZE_H - 1
        I = np.arange(lenI) * self.GRID_SIZE_W + np.full(lenI, self.GRID_SIZE_W / 2)
        J = np.arange(lenJ) * self.GRID_SIZE_H + np.full(lenJ, self.GRID_SIZE_H / 2)
        self.points1 = np.asarray([[t] for t in itertools.product(I, J)])
        self.imgPrevGray = imgGray.copy()
        self.prev_pyramid, self.pyramid = self.pyramid, self.prev_pyramid
        self.points0, self.points1 = self.points1, self.points0
