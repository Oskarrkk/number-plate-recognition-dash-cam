from detect_peaks import detect_peaks
import cv2
import numpy as np
import math
from scipy import ndimage

class Segmentation2():

    def segment(self, plate):
        treshold = self.process(plate)
        tab = self.filter_tab(treshold)
        chars = self.find_chars(tab)

        return chars

    def process(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        maxx = self.maximizeContrast(gray)
        imgBlurred2 = cv2.GaussianBlur(maxx, (3, 3), 0)
        img_edges = cv2.Canny(imgBlurred2, 100, 100, apertureSize=3)
        median_angle = self.compute_skew(img_edges)
        img_rotated = ndimage.rotate(maxx, median_angle)
        imgBlurred = cv2.GaussianBlur(img_rotated, (3, 3), 0)
        kernel_ver = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        vertical = cv2.filter2D(imgBlurred, -1, kernel_ver)

        m = 100  # np.max(proj)
        w = 1
        tab = []
        result2 = np.zeros((imgBlurred.shape[0], 300))
        for y in range(0, imgBlurred.shape[0], 1):
            maxa = 0
            piece1 = vertical[y, 0:imgBlurred.shape[1]]
            proj = np.sum(piece1, 0)
            suma = proj * w / m
            tab.append(int(suma))

        box = np.ones(3) / 3
        y2 = []
        y_smooth = np.convolve(tab, box, mode='same')

        for id in range(0, y_smooth.shape[0]):
            cv2.line(result2, (0, id), (int(y_smooth[id]), id), (255, 255, 255), 1)
            y2.append(y_smooth[id])
        peaks = detect_peaks(y2, mph=0.8 * max(y_smooth), mpd=50)
        yb0, yb1, result2 = self.filter_peaks(result2, peaks, y_smooth)

        tablica = img_rotated[yb0:yb1, 0:imgBlurred.shape[1]]
        tablica = cv2.threshold(tablica, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        tablica = cv2.medianBlur(tablica,3)
        return tablica

    def filter_tab(self, tablica):
        tablica = 255 - tablica

        nlabel4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(tablica,
                                                                                connectivity=8)

        for l in range(1, nlabel4):
            # if stats4[l, cv2.CC_STAT_HEIGHT] <= 0.7*tablica.shape[0]: #or stats4[l, cv2.CC_STAT_WIDTH] <= 10 or stats4[l, cv2.CC_STAT_WIDTH] >= tablica.shape[1] / 5:
            dst = tablica.copy()[stats4[l][1]:stats4[l][1] + stats4[l][3], stats4[l][0]:stats4[l][0] + stats4[l][2]]

            tablica[labels4 == l] = 0

            w = stats4[l][2]
            h = stats4[l][3]
            # cv2.waitKey(0)
            px = dst.shape[0] * dst.shape[1]
            max_white = 0.8 * px
            white = cv2.countNonZero(dst)
            black = px - white
            ratiowh = white / px
            ratio = h / w

            if white < max_white and h > 0.7 * tablica.shape[0] and ratio > 1.2 and ratio < 5:  # or px > 100:
                tablica[labels4 == l] = 255
            return tablica

    def find_chars(self, tablica):
        chars = []
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(tablica,
                                                                            connectivity=4)
        stats = sorted(stats, key=lambda stats: stats[0])
        for el in range(1, nlabel):
            dst = tablica[stats[el][1]:stats[el][1] + stats[el][3], stats[el][0]:stats[el][0] + stats[el][2]]
            w = stats[el][2]
            h = stats[el][3]
            px = dst.shape[0] * dst.shape[1]
            white = cv2.countNonZero(dst)
            black = px - cv2.countNonZero(dst)
            ratiobl = black / px
            ratiowh = white / px
            ratio = h / w
            char = None
            if ratiowh > 0.21 and ratiowh < 0.95 and ratio > 1.2 and ratio < 7.7 and px > 0.3 * tablica.shape[0] * w:
                char = cv2.resize(dst, (15, 20))
            if char is not None:
                chars.append(char)
        return chars

    def filter_peaks(self, result2, peaks, y_smooth):
        l = []
        for id in range(0, peaks.shape[0]):
            yb0 = 0
            yb1 = 0
            # cv2.line(result2, (0, peaks[id]), (int(y_smooth[peaks[id]] * 2), peaks[id]), (255, 0, 0), 2)
            val = y_smooth[peaks[id]] * 0.5
            for y in range(peaks[id] + 1, y_smooth.shape[0]):
                if y_smooth[y] < val:
                    yb1 = y
                    break

            for y in range(peaks[id] - 1, 0, -1):
                print(y_smooth[y])
                if y_smooth[y] < val:
                    yb0 = y
                    break

            #   if yb1 - yb0 < 50 and yb1 - yb0 > 2:
            cv2.line(result2, (0, yb0), (int(y_smooth[yb0] * 2), yb0), (255, 0, 0), 1)
            cv2.line(result2, (0, yb1), (int(y_smooth[yb1] * 2), yb1), (255, 0, 0), 1)
            hei = np.array([yb0, yb1])
            # l.append(hei)
            return yb0, yb1, result2

    def maximizeContrast(self,imgGrayscale):

        height, width = imgGrayscale.shape

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

    def compute_skew(self,image):
        height, width = image.shape
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength=width/1.5 , maxLineGap=20)
        angle = 0.0
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            median_angle = np.median(angles)
            return median_angle
            #img_rotated = ndimage.rotate(img_before, median_angle)

        return angle



