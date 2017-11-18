import numpy as np
import cv2
import ProbModel

# cap = cv2.VideoCapture('woman.mp4')
#
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# while(1):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#
#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
gray1 = cv2.imread('/home/mahdi/Apps/others/probModel/gg1.jpg', 2)
gray2 = cv2.imread('/home/mahdi/Apps/others/probModel/gg2.jpg', 2)
np.set_printoptions(precision=2)
model = ProbModel.ProbModel()
model.init(gray1)
h = np.asarray([0.19658485, -0.14968985, -0.00128087, 0.45442999, 0.18668709, 0.07489325, 0.07402678, -0.48996603, 0.3031477])\
    .reshape(3,3)
model.motionCompensate(h)
model.update()
