import numpy as np
import cv2
import itertools

class ProbModel:
    def __init__(self):
        self.NUM_MODELS = 2
        self.BLOCK_SIZE	= 4
        self.INIT_BG_VAR = 25 * 25
        self.VAR_THRESH_MODEL_MATCH = 2
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

    def rebin(self, arr, factor):
        temp = np.pad(arr, ((0,arr.shape[0] % factor[0]), (0, arr.shape[1] % factor[1])), 'edge')
        sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
        return temp.reshape(sh).mean(-1).mean(1)

    def motionCompensate(self, H):

        I = np.asarray(range(self.modelWidth) * self.modelHeight)
        J = np.repeat(range(self.modelHeight), self.modelWidth)


        points = np.asarray([I*self.BLOCK_SIZE+self.BLOCK_SIZE/2, J*self.BLOCK_SIZE + self.BLOCK_SIZE/2, np.ones(len(I))])

        tempMean = H.dot(points)
        NewW = tempMean[2, :]
        NewX = (tempMean[0, :]/NewW)
        NewY = (tempMean[1, :]/NewW)

        NewI = NewX / self.BLOCK_SIZE
        NewJ = NewY / self.BLOCK_SIZE

        idxNewI = np.floor(NewI).astype(int)
        idxNewJ = np.floor(NewJ).astype(int)

        Di = NewI - idxNewI - 0.5
        Dj = NewJ - idxNewJ - 0.5

        aDi = abs(Di)
        aDj = abs(Dj)

        M = self.means[0]
        V = self.vars[0]
        A = self.ages[0]

        W_H = (aDi * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)
        W_V = (aDj * (1 - aDi)).reshape(self.modelHeight, self.modelWidth)
        W_HV = (aDi * aDj).reshape(self.modelHeight, self.modelWidth)
        W_self = ((1-aDi) * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)

        W = np.zeros((self.modelHeight, self.modelWidth))

        tempMean = np.zeros(self.means[0].shape)
        tempAges = np.zeros(self.means[0].shape)


        NewI_H = idxNewI + np.sign(Di).astype(int)
        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        tempMean[J[condH], I[condH]] = W_H[J[condH], I[condH]] * M[idxNewJ[condH], NewI_H[condH]]
        tempAges[J[condH], I[condH]] = W_H[J[condH], I[condH]] * A[idxNewJ[condH], NewI_H[condH]]
        W[J[condH], I[condH]] += W_H[J[condH], I[condH]]

        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[J[condV], I[condV]] += W_V[J[condV], I[condV]] * M[NewJ_V[condV], idxNewI[condV]]
        tempAges[J[condV], I[condV]] += W_V[J[condV], I[condV]] * A[NewJ_V[condV], idxNewI[condV]]
        W[J[condV], I[condV]] += W_V[J[condV], I[condV]]

        NewI_H = idxNewI + np.sign(Di).astype(int)
        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condHV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        tempMean[J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * M[NewJ_V[condHV], NewI_H[condHV]]
        tempAges[J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * A[NewJ_V[condHV], NewI_H[condHV]]
        W[J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]]

        condSelf = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * M[idxNewJ[condSelf], idxNewI[condSelf]]
        tempAges[J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * A[idxNewJ[condSelf], idxNewI[condSelf]]
        W[J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]]

        self.temp_means[0][W != 0] = 0
        W[W == 0] = 1
        self.temp_means[0] += tempMean / W
        self.temp_means[1] += tempAges / W

        temp_var = np.zeros(self.means[0].shape)

        temp_var[J[condH], I[condH]] += W_H[J[condH], I[condH]] * (V[idxNewJ[condH], NewI_H[condH]] +
                                                                  np.power(self.temp_means[0][J[condH], I[condH]] -
                                                                           self.means[0][idxNewJ[condH], NewI_H[condH]],
                                                                           2))

        temp_var[J[condV], I[condV]] += W_V[J[condV], I[condV]] * (V[NewJ_V[condV], idxNewI[condV]] +
                                                                   np.power(self.temp_means[0][J[condV], I[condV]] -
                                                                            self.means[0][
                                                                                NewJ_V[condV], idxNewI[condV]],
                                                                            2))

        temp_var[J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * (V[NewJ_V[condHV], NewI_H[condHV]] +
                                                                  np.power(self.temp_means[0][J[condHV], I[condHV]] -
                                                                           self.means[0][NewJ_V[condHV], NewI_H[condHV]],
                                                                           2))

        temp_var[J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * (V[idxNewJ[condSelf], idxNewI[condSelf]] +
                                                                  np.power(self.temp_means[0][J[condSelf], I[condSelf]] -
                                                                           self.means[0][idxNewJ[condSelf], idxNewI[condSelf]],
                                                                           2))

        print (temp_var/W).reshape(self.modelHeight* self.modelWidth)


    def update(self):
        curMean = self.rebin(self.curImage, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        mm = np.argmax(self.temp_ages, axis=0).reshape(-1)
        maxes = np.max(self.temp_ages, axis=0)
        h, w = self.modelWidth * self.modelHeight
        jj, ii = np.arange(h*w)/w, np.arange(h*w)%w
        ii, jj = ii[mm != 0], jj[mm != 0]
        mm = mm[mm != 0]
        self.temp_ages[mm, jj, ii] = 0
        self.temp_ages[0] = maxes

        self.temp_means[0, jj, ii] = self.temp_means[mm, jj, ii]
        self.temp_means[mm, jj, ii] = curMean[jj, ii]

        self.temp_vars[0, jj, ii] = self.temp_vars[mm, jj, ii]
        self.temp_vars[mm, jj, ii] = self.INIT_BG_VAR

        modelIndex = np.ones(curMean.shape)
        cond1 = np.power(curMean - self.temp_means[0], 2) < self.VAR_THRESH_MODEL_MATCH * self.temp_vars[0]
        modelIndex[cond1] = 0
        cond2 = np.power(curMean - self.temp_means[1], 2) < self.VAR_THRESH_MODEL_MATCH * self.temp_vars[1]
        modelIndex[cond2] = 1
        self.temp_ages[1][(~cond1) & ~(cond2)] = 0



