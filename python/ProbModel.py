import numpy as np
import cv2
import itertools

class ProbModel:
    def __init__(self):
        self.NUM_MODELS = 2
        self.BLOCK_SIZE	= 4
        self.VAR_THRESH_MODEL_MATCH = 2
        self.MAX_BG_AGE = 30
        self.VAR_THRESH_FG_DETERMINE = 4.0
        self.INIT_BG_VAR = 20.0*20.0
        self.MIN_BG_VAR = 5 * 5
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
        (self.modelHeight, self.modelWidth) = (self.obsHeight//self.BLOCK_SIZE, self.obsWidth//self.BLOCK_SIZE)
        self.means = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.vars = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.ages = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))

        self.modelIndexes = np.zeros((self.modelWidth, self.modelHeight))
        self.means =  np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.temp_means = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.temp_ages = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        self.temp_vars = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))
        H = np.identity(3)
        self.motionCompensate(H)
        self.update(gray)

    def rebin(self, arr, factor):
        f = (np.asarray(factor) - arr.shape) % factor
        temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
        sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
        res = temp.reshape(sh).mean(-1).mean(1)
        return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]

    def rebinMax(self, arr, factor):
        f = (np.asarray(factor) - arr.shape) % factor
        temp = np.pad(arr, ((0, f[0]), (0, f[1])), 'edge')
        sh = temp.shape[0] // factor[0], factor[0], -1, factor[1]
        res = temp.reshape(sh).max(-1).max(1)
        return res[:res.shape[0] - f[0], : res.shape[1] - f[1]]

    def motionCompensate(self, H):

        I = np.array([range(self.modelWidth)]*self.modelHeight).flatten()
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

        M = self.means
        V = self.vars
        A = self.ages

        W_H = (aDi * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)
        W_V = (aDj * (1 - aDi)).reshape(self.modelHeight, self.modelWidth)
        W_HV = (aDi * aDj).reshape(self.modelHeight, self.modelWidth)
        W_self = ((1-aDi) * (1 - aDj)).reshape(self.modelHeight, self.modelWidth)

        W = np.zeros((self.NUM_MODELS, self.modelHeight, self.modelWidth))

        tempMean = np.zeros(self.means.shape)
        tempAges = np.zeros(self.means.shape)


        NewI_H = idxNewI + np.sign(Di).astype(int)
        condH = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)

        tempMean[:, J[condH], I[condH]] = W_H[J[condH], I[condH]] * M[:, idxNewJ[condH], NewI_H[condH]]
        tempAges[:, J[condH], I[condH]] = W_H[J[condH], I[condH]] * A[:, idxNewJ[condH], NewI_H[condH]]
        W[:, J[condH], I[condH]] += W_H[J[condH], I[condH]]

        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[:, J[condV], I[condV]] += W_V[J[condV], I[condV]] * M[:, NewJ_V[condV], idxNewI[condV]]
        tempAges[:, J[condV], I[condV]] += W_V[J[condV], I[condV]] * A[:, NewJ_V[condV], idxNewI[condV]]
        W[:, J[condV], I[condV]] += W_V[J[condV], I[condV]]

        NewI_H = idxNewI + np.sign(Di).astype(int)
        NewJ_V = idxNewJ + np.sign(Dj).astype(int)
        condHV = (NewJ_V >= 0) & (NewJ_V < self.modelHeight) & (NewI_H >= 0) & (NewI_H < self.modelWidth)
        tempMean[:, J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * M[:, NewJ_V[condHV], NewI_H[condHV]]
        tempAges[:, J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * A[:, NewJ_V[condHV], NewI_H[condHV]]
        W[:, J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]]

        condSelf = (idxNewJ >= 0) & (idxNewJ < self.modelHeight) & (idxNewI >= 0) & (idxNewI < self.modelWidth)
        tempMean[:, J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * M[:, idxNewJ[condSelf], idxNewI[condSelf]]
        tempAges[:, J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * A[:, idxNewJ[condSelf], idxNewI[condSelf]]
        W[:, J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]]

        self.temp_means[W != 0] = 0

        self.temp_ages[:] = 0
        W[W == 0] = 1
        self.temp_means += tempMean / W
        self.temp_ages += tempAges / W

        temp_var = np.zeros(self.means.shape)

        temp_var[:, J[condH], I[condH]] += W_H[J[condH], I[condH]] * (V[:, idxNewJ[condH], NewI_H[condH]] +
                                                                  np.power(self.temp_means[:, J[condH], I[condH]] -
                                                                           self.means[:, idxNewJ[condH], NewI_H[condH]],
                                                                           2))

        temp_var[:, J[condV], I[condV]] += W_V[J[condV], I[condV]] * (V[:, NewJ_V[condV], idxNewI[condV]] +
                                                                   np.power(self.temp_means[:, J[condV], I[condV]] -
                                                                            self.means[:,
                                                                                NewJ_V[condV], idxNewI[condV]],
                                                                            2))

        temp_var[:, J[condHV], I[condHV]] += W_HV[J[condHV], I[condHV]] * (V[:, NewJ_V[condHV], NewI_H[condHV]] +
                                                                  np.power(self.temp_means[:, J[condHV], I[condHV]] -
                                                                           self.means[:, NewJ_V[condHV], NewI_H[condHV]],
                                                                           2))

        temp_var[:, J[condSelf], I[condSelf]] += W_self[J[condSelf], I[condSelf]] * (V[:, idxNewJ[condSelf], idxNewI[condSelf]] +
                                                                  np.power(self.temp_means[:, J[condSelf], I[condSelf]] -
                                                                           self.means[:, idxNewJ[condSelf], idxNewI[condSelf]],
                                                                           2))


        self.temp_vars = temp_var / W
        cond = (idxNewJ < 1) | (idxNewJ >= self.modelHeight - 1) | (idxNewI < 1) | (idxNewI >= self.modelWidth - 1)
        self.temp_vars[:, J[cond], I[cond]] = self.INIT_BG_VAR
        self.temp_ages[:, J[cond], I[cond]] = 0
        self.temp_vars[self.temp_vars < self.MIN_BG_VAR] = self.MIN_BG_VAR




    def update(self, gray):
        curMean = self.rebin(gray, (self.BLOCK_SIZE, self.BLOCK_SIZE))
        mm = self.NUM_MODELS - np.argmax(self.temp_ages[::-1], axis=0).reshape(-1) - 1
        maxes = np.max(self.temp_ages, axis=0)
        h, w = self.modelHeight , self.modelWidth
        jj, ii = np.arange(h*w)//w, np.arange(h*w)%w

        ii, jj = ii[mm != 0], jj[mm != 0]
        mm = mm[mm != 0]
        self.temp_ages[mm, jj, ii] = 0
        self.temp_ages[0] = maxes

        self.temp_means[0, jj, ii] = self.temp_means[mm, jj, ii]
        self.temp_means[mm, jj, ii] = curMean[jj, ii]

        self.temp_vars[0, jj, ii] = self.temp_vars[mm, jj, ii]
        self.temp_vars[mm, jj, ii] = self.INIT_BG_VAR

        modelIndex = np.ones(curMean.shape).astype(int)
        cond1 = np.power(curMean - self.temp_means[0], 2) < self.VAR_THRESH_MODEL_MATCH * self.temp_vars[0]

        cond2 = np.power(curMean - self.temp_means[1], 2) < self.VAR_THRESH_MODEL_MATCH * self.temp_vars[1]
        modelIndex[cond1] = 0
        modelIndex[cond2 & ~cond1] = 1
        self.temp_ages[1][(~cond1) & (~cond2)] = 0



        modelIndexMask = np.arange(self.means.shape[0]).reshape(-1, 1, 1) == modelIndex

        alpha = self.temp_ages / (self.temp_ages + 1)
        alpha[self.temp_ages < 1] = 0
        alpha[~modelIndexMask] = 1
        self.means = self.temp_means * alpha + curMean * (1 - alpha)

        jj, ii = np.arange(h * w) // w, np.arange(h * w) % w

        bigMeanIndex = np.kron(self.means[modelIndex.reshape(-1), jj, ii].reshape(h, -1), np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        bigMean = np.kron(self.means[0], np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        bigAges = np.kron(self.ages[0], np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        bigVars = np.kron(self.vars[0], np.ones((self.BLOCK_SIZE, self.BLOCK_SIZE)))
        (a, b) = (gray.shape[0] - bigMean.shape[0], gray.shape[1] - bigMean.shape[1])
        bigMean = np.pad(bigMean, ((0, a), (0, b)), 'edge')
        bigAges = np.pad(bigAges, ((0, a), (0, b)), 'edge')
        bigVars = np.pad(bigVars, ((0, a), (0, b)), 'edge')
        bigMeanIndex = np.pad(bigMeanIndex, ((0, a), (0, b)), 'edge')

        maxes = self.rebinMax(np.power(gray - bigMeanIndex, 2), (self.BLOCK_SIZE, self.BLOCK_SIZE))
        self.distImg = np.power(gray - bigMean, 2)
        out = np.zeros(gray.shape).astype(np.uint8)
        out[(bigAges > 1) & (self.distImg > self.VAR_THRESH_FG_DETERMINE * bigVars)] = 255
		
        alpha = self.temp_ages / (self.temp_ages + 1)
        alpha[~modelIndexMask] = 1

        self.vars = self.temp_vars * alpha + (1 - alpha) * maxes

        self.vars[(self.vars < self.INIT_BG_VAR) & modelIndexMask & (self.ages == 0)] = self.INIT_BG_VAR
        self.vars[(self.vars < self.MIN_BG_VAR) & modelIndexMask] = self.MIN_BG_VAR

        self.ages = self.temp_ages.copy()
        self.ages[modelIndexMask] += 1
        self.ages[modelIndexMask & (self.ages > 30)] = 30

        return out
