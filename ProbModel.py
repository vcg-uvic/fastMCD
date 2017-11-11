import numpy as np
import cv2
import itertools

class ProbModel:
    def __init__(self):
        self.NUM_MODELS = 2
        self.BLOCK_SIZE	= 4.0
        self.curImage = None
        self.means = None
        self.vars = None
        self.ages = None
        self.temp_means = None
        self.temp_vars = None
        self.temp_ages = None

        self.modelIndexes = None
        self.modelWidth = None
        self.modelHeight = None
        self.obsWidth = None
        self.obsHeight = None


    def init(self, gray):
        (self.obsHeight, self.obsWidth, d) = gray.shape
        self.curImage = gray
        (self.modelHeight, self.modelWidth) = (self.obsHeight/self.BLOCK_SIZE, self.obsWidth/self.BLOCK_SIZE)
        self.means = np.zeros((self.NUM_MODELS, self.modelWidth*self.modelHeight))
        self.vars = np.zeros((self.NUM_MODELS, self.modelWidth*self.modelHeight))
        self.ages = np.zeros((self.NUM_MODELS, self.modelWidth*self.modelHeight))
        self.modelIndexes = np.zeros(self.modelWidth*self.modelHeight)

        h = np.identity(3)

    def shift(self, arr, rr, cc):
        (r, c) = arr.shape
        t1 = (abs(rr), abs(rr))
        t2 = (abs(cc), abs(cc))
        (a, b) = (0, -2 * rr) if rr > 0 else (2 * abs(rr), r + 2 * abs(rr))
        (c, d) = (0, -2 * cc) if cc > 0 else (2 * abs(cc), c + 2 * abs(cc))
        return np.pad(arr, (t1, t2), 'constant')[a:b, c:d]

    def motionCompensate(self, H):


        I = np.arange(self.modelWidth) * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        J = np.arange(self.modelHeight) * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        points = np.asarray(list(itertools.product(I, J, [1]))).transpose()
        temp = H.dot(points)
        NewW = temp[2:]
        NewX = (temp[0:]/NewW).reshape(self.modelHeight, self.modelWidth)
        NewY = (temp[1:]/NewW).reshape(self.modelHeight, self.modelWidth)

        NewI = NewX / self.BLOCK_SIZE
        NewJ = NewY / self.BLOCK_SIZE

        Di = NewI - np.floor(NewI) - 0.5
        Dj = NewJ - np.floor(NewJ) - 0.5
        aDi = abs(Di)
        aDj = abs(Dj)

        W_H = abs(Di) * (1 - abs(Dj))

        leftMean = self.shift(self.means, 0, -1)
        rightMean = self.shift(self.means, 0, 1)
        W_H * ((Di+aDi)*leftMean + (aDi - Di)*rightMean) / 2