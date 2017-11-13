import numpy as np
import cv2
import itertools

class ProbModel:
    def __init__(self):
        self.NUM_MODELS = 2
        self.BLOCK_SIZE	= 4
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
        (self.obsHeight, self.obsWidth) = gray.shape
        self.curImage = gray
        (self.modelHeight, self.modelWidth) = (self.obsHeight/self.BLOCK_SIZE, self.obsWidth/self.BLOCK_SIZE)
        self.means = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.vars = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.ages = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.modelIndexes = np.zeros((self.modelWidth, self.modelHeight))
        self.means = np.arange(self.NUM_MODELS * self.modelHeight * self.modelWidth).reshape(self.NUM_MODELS, self.modelHeight, self.modelWidth).astype(float)
        self.temp_means = -1 * np.ones((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        h = np.identity(3)

    def shift(self, arr, rr, cc):
        (r, c) = arr.shape
        t1 = (abs(rr), abs(rr))
        t2 = (abs(cc), abs(cc))
        (a, b) = (0, -2 * rr) if rr > 0 else (2 * abs(rr), r + 2 * abs(rr))
        (c, d) = (0, -2 * cc) if cc > 0 else (2 * abs(cc), c + 2 * abs(cc))
        return np.pad(arr, (t1, t2), 'constant')[a:b, c:d]

    def motionCompensate(self, H):

        I = np.asarray(range(self.modelWidth) * self.modelHeight)
        J = np.repeat(range(self.modelHeight), self.modelWidth)


        points = np.asarray([I*self.BLOCK_SIZE+self.BLOCK_SIZE/2, J*self.BLOCK_SIZE + self.BLOCK_SIZE/2, np.ones(len(I))])

        temp = H.dot(points)
        NewW = temp[2, :]
        NewX = (temp[0, :]/NewW)
        NewY = (temp[1, :]/NewW)

        NewI = NewX / self.BLOCK_SIZE
        NewJ = NewY / self.BLOCK_SIZE

        idxNewI = np.floor(NewI).astype(int)
        idxNewJ = np.floor(NewJ).astype(int)

        Di = NewI - idxNewI - 0.5
        Dj = NewJ - idxNewJ - 0.5

        aDi = abs(Di)
        aDj = abs(Dj)

        m = self.means[0]
        v = self.vars[0]

        W_H = (aDi * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)
        W_V = (aDj * (1 - aDi)).reshape(self.modelHeight, self.modelWidth)
        W_HV = (aDi * aDj).reshape(self.modelHeight, self.modelWidth)
        W_self = ((1-aDi) * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)

        W = np.zeros((self.modelHeight, self.modelWidth))

        temp = np.zeros(self.means[0].shape)


        NewI_H = idxNewI + np.sign(Di).astype(int)
        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        temp[J[condH], I[condH]] = W_H[J[condH], I[condH]] * m[idxNewJ[condH], NewI_H[condH]]
        W[J[condH], I[condH]] += W_H[J[condH], I[condH]]

        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condH = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        temp[J[condH], I[condH]] += W_V[J[condH], I[condH]] * m[NewJ_V[condH], idxNewI[condH]]
        W[J[condH], I[condH]] += W_V[J[condH], I[condH]]

        NewI_H = idxNewI + np.sign(Di).astype(int)
        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condH = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        temp[J[condH], I[condH]] += W_HV[J[condH], I[condH]] * m[NewJ_V[condH], NewI_H[condH]]
        W[J[condH], I[condH]] += W_HV[J[condH], I[condH]]

        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        temp[J[condH], I[condH]] += W_self[J[condH], I[condH]] * m[idxNewJ[condH], idxNewI[condH]]
        W[J[condH], I[condH]] += W_self[J[condH], I[condH]]

        self.temp_means[0][W != 0] = 0
        W[W == 0] = 1
        self.temp_means[0] += temp/W

        temp_var = np.zeros(self.means[0].shape)

        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        temp_var[J[condH], I[condH]] = W_H[J[condH], I[condH]] * (v[idxNewJ[condH], NewI_H[condH]] +
                                                                  np.power(self.temp_means[0][J[condH], I[condH]] -
                                                                                self.means[0][idxNewJ[condH], NewI_H[condH]], 2))



        print temp_var.reshape(self.modelHeight* self.modelWidth)


