import numpy as np
import cv2
import MCDWrapper

np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture('woman.mp4')
mcd = MCDWrapper.MCDWrapper()
isFirst = True
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if (isFirst):
        mcd.init(gray)
        isFirst = False
    else:
        mask = mcd.run(gray)
    frame[mask > 0, 2] = 255
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

