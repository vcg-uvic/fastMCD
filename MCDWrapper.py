import numpy as np
import cv2



class MCDWrapper:
    def __init__(self):
        self.imgIpl = None
        self.imgGray = None
        self.imgGrayPrev = None
        self.frm_cnt = 0


    def run(self, frame):
        self.frm_cnt += 1
        self.imgIpl = frame
        imgGray = cv2.cvtColor(self.imgIpl, cv2.COLOR_BGR2GRAY)
        self.imgGray = cv2.medianBlur(imgGray, 5)
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()
        








