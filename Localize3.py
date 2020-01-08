import cv2
import numpy as np


class DetectorMorph():

    def pre_process(self,frame):
        height, width, numChannels = frame.shape

        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgGrayscale = cv2.split(imgHSV)

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        imgBlurred = np.zeros((height, width, 1), np.uint8)
        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (3, 3), 0)

        kernel_ver = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        vertical = cv2.filter2D(imgBlurred, -1, kernel_ver)

        th4 = cv2.threshold(imgTopHat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


        return th4,vertical,imgTopHat

    def filter_contour(self,threshold):
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 7)))
        nlabel4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(threshold,
                                                                                connectivity=8)
        img2 = np.zeros(threshold.shape, np.uint8)
        for l in range(1, nlabel4):
            ratio = stats4[l, cv2.CC_STAT_WIDTH] / stats4[l, cv2.CC_STAT_HEIGHT]
            if ratio > 3 and ratio < 7 and stats4[l, cv2.CC_STAT_HEIGHT] < 50 and stats4[l, cv2.CC_STAT_AREA] > 50:
                img2[labels4 == l] = int(255)

        return img2

    def find_plate(self,frame):

        candidates = []
        xy_candidates = []
        threshold, vertical, imgTopHat= self.pre_process(frame)
        area = self.filter_contour(threshold)

        nlabel2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(area, connectivity=8)
        for el in range(1, nlabel2):

            if stats2[el][1] - 3 >0:
                stats2[el][1] = stats2[el][1] - 3
            if stats2[el][0] - 3 >0:
                stats2[el][0] = stats2[el][0] - 3
            if stats2[el][1] + stats2[el][3] + 3 < vertical.shape[0]:
                stats2[el][3] = stats2[el][3] + 3
            if stats2[el][0] + stats2[el][2] +3 < vertical.shape[1]:
                stats2[el][2] = stats2[el][2] + 3
            candidates.append(frame[stats2[el][1] :stats2[el][1] + stats2[el][3],
            stats2[el][0]:stats2[el][0] + stats2[el][2]])
            xy_candidates.append([stats2[el][1],stats2[el][0],stats2[el][1] + stats2[el][3],stats2[el][0] + stats2[el][2]])

        return candidates, xy_candidates

