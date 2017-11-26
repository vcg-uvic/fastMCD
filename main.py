import numpy as np
import cv2
import MCDWrapper

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
np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture('woman.mp4')
mcd = MCDWrapper.MCDWrapper()
if (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[0]/8, frame.shape[1]/8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mcd.init(gray)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[0] / 8, frame.shape[1] / 8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print frame.shape
    mask = mcd.run(gray)
    cv2.imshow('frame', gray * (mask/255) )
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

#h = np.asarray([0.19658485, -0.14968985, -0.00128087, 0.45442999, 0.18668709, 0.07489325, 0.07402678, -0.48996603, 0.3031477])\
#    .reshape(3,3)
