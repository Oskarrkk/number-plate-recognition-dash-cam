
import cv2
import numpy as np


class Segmentation():

    def segment(self,plate):
        threshold = self.process(plate)
        chars = self.excract(threshold)

        return chars


    def maximizeContrast(self,imgGrayscale):

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

    def compute_skew(self,image):
        kernel_hor = np.array([[ -1, -2, -1], [0, 0, 0], [1, 2, 1]])
        image = self.maximizeContrast(image)
        image = cv2.GaussianBlur(image, (5,5), 0)
        image = cv2.filter2D(image, -1, kernel_hor)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        height, width = image.shape
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength=width / 3.0, maxLineGap=20)
        angle = 0.0

        if lines is not None:
            nlines = lines.shape[0]
            lines = lines.reshape(lines.shape[0], 4)
            for x1, y1, x2, y2 in lines:
                angle += np.arctan2(y2 - y1, x2 - x1)
            angle /= nlines

            return angle * 180 / np.pi
        return angle

    def deskew2(self,image, angle):
        (h, w) = image.shape[:2]
        non_zero_pixels = cv2.findNonZero(image)
        center = (w / 2, h / 2)

        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = image.shape
        rotated1 = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC, borderValue=(127,127,127))

        return cv2.getRectSubPix(rotated1, (cols, rows), center)


    def process(self,plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        maxx = self.maximizeContrast(gray)
        deskewed_image3 = self.deskew2(maxx.copy(), self.compute_skew(maxx))

        th5 = cv2.threshold(deskewed_image3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        th5 = th5 - 255
        return th5


    def excract(self, th5):
        nlabel4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(th5,
                                                                                connectivity=8)
        for l in range(1, nlabel4):
            if stats4[l, cv2.CC_STAT_HEIGHT] >= th5.shape[0] / 4 and stats4[l, cv2.CC_STAT_WIDTH] >= 5 and stats4[
                l, cv2.CC_STAT_WIDTH] <= th5.shape[0] / 2:
                th5[labels4 == l] = 255

        chars = []
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(th5,
                                                                         connectivity=4)
        stats = sorted(stats, key=lambda stats: stats[0])
        for el in range(1, nlabel):
            dst = th5[stats[el][1]:stats[el][1] + stats[el][3], stats[el][0]:stats[el][0] + stats[el][2]]
            w = stats[el][2]
            h = stats[el][3]
            px = dst.shape[0] * dst.shape[1]
            white = cv2.countNonZero(dst)
            ratiowh = white / px
            ratio = h / w
            char = None

            if ratiowh > 0.21 and ratiowh < 0.95 and ratio > 1.2 and ratio < 7.7 and px > 0.3 * th5.shape[0] * w:
                char = cv2.resize(dst, (15,20))
            if char is not None:
                chars.append(char)
        return chars


