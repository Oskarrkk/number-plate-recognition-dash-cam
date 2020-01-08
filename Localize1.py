import cv2
import numpy as np
from detect_peaks import detect_peaks

class DetectorEdges():

    def pre_process(self,frame):
        height, width, numChannels = frame.shape

        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgGrayscale = cv2.split(imgHSV)

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)


        return imgMaxContrastGrayscale

    def filter_peaks(self,result2, peaks, y_smooth):
        l = []
        for id in range(0, peaks.shape[0]):
            yb0 = 0
            yb1 = 0
            cv2.line(result2, (0, peaks[id]), (int(y_smooth[peaks[id]] * 1.3), peaks[id]), (255, 0, 0), 1)
            val = y_smooth[peaks[id]] * 0.55
            for y in range(peaks[id] + 1, y_smooth.shape[0]):
                if y_smooth[y] < val:
                    yb1 = y
                    break

            for y in range(peaks[id] - 1, 0, -1):
                if y_smooth[y] < val:
                    yb0 = y
                    break

            if yb1 - yb0 < 50 and yb1 - yb0 > 2:
                cv2.line(result2, (0, yb0), (int(y_smooth[yb0] * 2), yb0), (255, 0, 0), 1)
                cv2.line(result2, (0, yb1), (int(y_smooth[yb1] * 2), yb1), (255, 0, 0), 1)
                hei = np.array([yb0, yb1])
                l.append(hei)
        return l, result2

    def vertical_projection(self,imgMaxContrast):
        kernel1 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]) / 7
        kernel_ver = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        rozmazane = cv2.filter2D(imgMaxContrast, -1, kernel1)
        vertical = cv2.filter2D(rozmazane, -1, kernel_ver)
        result2 = np.zeros((vertical.shape[0], 300))
        tab = []

        m = 255
        w = 10
        wh = 15
        ww = wh * 4
        stepSizeW = int(0.5 * ww)
        windowSize = [ww, wh]

        for y in range(0, vertical.shape[0], 1):
            maxa = 0
            #   print(y)

            for x in range(0, vertical.shape[1], stepSizeW):
                piece1 = vertical[y, x:x + windowSize[0]]
                proj = np.sum(piece1, 0)

                suma = proj * w / m
                if suma > maxa:
                    maxa = suma
            tab.append(int(maxa))

        box = np.ones(7) / 7
        y2 = []
        y_smooth = np.convolve(tab, box, mode='same')
        for id in range(0, y_smooth.shape[0]):
            cv2.line(result2, (0, id), (int(y_smooth[id]), id), (255, 255, 255), 1)
            y2.append(y_smooth[id])
        peaks = detect_peaks(y2, mph=0.50 * max(y_smooth), mpd=50)


        return peaks, y_smooth, result2

    def find_horizontal_pos(self,peaks2, x_smooth, result1):
        l1 = []
        for id3 in range(0, peaks2.shape[0]):

            xp0 = 0
            xp1 = 0
            cv2.line(result1, (peaks2[id3], 0), (peaks2[id3], int(x_smooth[peaks2[id3]] * 2)), (255, 0, 0), 2)
            valx = x_smooth[peaks2[id3]] * 0.4
            for x2 in range(peaks2[id3] + 1, x_smooth.shape[0]):
                if x_smooth[x2] < valx:
                    xp1 = x2
                    break

            for x2 in range(peaks2[id3] - 1, 0, -1):
                if x_smooth[x2] < valx:
                    xp0 = x2
                    break

            if xp0 > 0 and xp1 > 0:
                whi = np.array([xp0, xp1])
                l1.append(whi)
                cv2.line(result1, (xp0, 0), (xp0, int(x_smooth[xp0] * 10)), (255, 0, 0), 1)
                cv2.line(result1, (xp1, 0), (xp1, int(x_smooth[xp1] * 10)), (255, 0, 0), 1)

        return l1, result1


    def horizontal_projection(self, imgMaxContrast, l,frame):

        kernel_hor = np.array([[ -1, -2, -1], [ 0, 0, 0], [1, 2, 1]])
        kernel2 = np.array([1, 1, 1]) / 3
        rozmazane2 = cv2.filter2D(imgMaxContrast, -1, kernel2)
        horizontal = cv2.filter2D(rozmazane2, -1, kernel_hor)
        peaks2=[]
        candidates = []
        xy_candidates = []
        result1 = np.zeros((300, horizontal.shape[1]))
        print(len(l))
        for id2 in range(0, len(l)):
            result1 = np.zeros((200, horizontal.shape[1]))
            box2 = np.ones(int((l[id2][1] - l[id2][0]) / 2)) / int((l[id2][1] - l[id2][0]) / 2)
            dst = cv2.reduce(horizontal[l[id2][0]:l[id2][1], 0:horizontal.shape[1]], 0, cv2.REDUCE_SUM,
                             dtype=cv2.CV_32S)
            m1 = np.max(dst)
            w1 = 100
            result1 = np.zeros((200, dst.shape[1]))
            xx2 = []
            tab1 = []
            for x1 in range(0, horizontal.shape[1], 1):
                tab1.append(dst[0][x1] * w1 / m1)
            x_smooth = np.convolve(tab1, box2, mode='same')

            for x1 in range(0, x_smooth.shape[0]):
                cv2.line(result1, (x1, 0), (x1, int(x_smooth[x1])), (255, 255, 255), 1)
                xx2.append(x_smooth[x1])
            peaks2 = detect_peaks(xx2, mph=0.75 * max(x_smooth), mpd=50)
            list_of_peaks_horizontal, result1 = self.find_horizontal_pos(peaks2, x_smooth, result1)
            if (list_of_peaks_horizontal):

                for jj in range(0, len(list_of_peaks_horizontal)):
                    candidates.append(frame[ l[id2][0] : l[id2][1], list_of_peaks_horizontal[jj][0] : list_of_peaks_horizontal[jj][1]])
                    xy_candidates.append([l[id2][0], list_of_peaks_horizontal[jj][0],l[id2][1] ,list_of_peaks_horizontal[jj][1]])

        return peaks2, result1,candidates,xy_candidates



    def find_plate(self,frame):
        candidates = []
        xy_candidates = []
        imgMaxContrast = self.pre_process(frame)
        peaks, y_smooth, result2 = self.vertical_projection(imgMaxContrast)
        list_of_peaks_vertical, result2 = self.filter_peaks(result2, peaks, y_smooth)

        if (list_of_peaks_vertical):
            peaks2, result1,candidates,xy_candidates = self.horizontal_projection(imgMaxContrast, list_of_peaks_vertical,frame)

        return candidates,xy_candidates

