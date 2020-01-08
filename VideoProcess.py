import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from imutils.video import FPS
import imutils
from skimage import feature
import Segmentation_chars
import Segmentation_chars2
import pickle
import Localize3
import Localize2
import Localize1
import Tracker
import Syntactic_analysator




class Video_Player(QtCore.QObject):
    pix = pyqtSignal(QPixmap)
    frame = pyqtSignal(np.ndarray)

    def __init__(self, dir, method,roi,method2, parent=None):
        super(Video_Player, self).__init__()
        self.video = cv2.VideoCapture(dir)
        self.last_frame = np.zeros((1, 1))
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.localize_method = method
        self.exctact_method = method2
        self.roi = roi
        self.count = 0
        self.stopped = True
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.flag = 0
        self.tracker = Tracker.Tracker(20)
        self.syntactic = Syntactic_analysator.Syntatic_analysator()
        filename = 'finalized_model6.sav'
        filename3 = 'ocr_model.sav'
        self.loaded_model = pickle.load(open(filename, 'rb'))
        self.ocr = pickle.load(open(filename3, 'rb'))
        print(self.loaded_model)
        self.class_names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8',
                   '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                   'L', 'M', 'N', 'O', 'P', 'R','S', 'T',  'U', 'V', 'W',
                   'X', 'Y', 'Z','']



    def get_frame(self):
        sift = cv2.xfeatures2d.SIFT_create()
        fps = FPS().start()
        if self.localize_method == 3:
            self.Detector = Localize2.DetectorMorph()
        if self.localize_method == 2:
            self.Detector = Localize1.DetectorEdges()
        if self.localize_method == 1:
            self.Detector = Localize3.DetectorMorph()
        if self.exctact_method == 1:
            self.Extractor = Segmentation_chars.Segmentation()
        if self.exctact_method == 2:
            self.Extractor = Segmentation_chars2.Segmentation2()

        j = 0
        while self.video.get(cv2.CAP_PROP_POS_FRAMES) < self.num_frames and self.stopped == False:
            #flag when plate is found on frame
            found = 0
            ret, self.last_frame  = self.video.read()
            frame = cv2.resize(self.last_frame, None, fx=0.5, fy=0.5)
            tablice = []
            numbers = []
            if self.roi == 1:
                frame_roi = frame[int(self.y*(frame.shape[0]/400)):int(self.y*(frame.shape[0]/400)+self.height*(frame.shape[0]/400)), int(self.x*(frame.shape[1]/630)):int(self.x*(frame.shape[1]/630)+self.width*(frame.shape[1]/630))]
                candidates,xy_candidates = self.Detector.find_plate(frame_roi)
            if self.roi == 2:
                candidates, xy_candidates = self.Detector.find_plate(frame)

            for i in range(0,len(candidates)):

                cand = self.last_frame[(xy_candidates[i][0]+int(self.y*(frame.shape[0]/400))) * 2: (xy_candidates[i][2]+int(self.y*(frame.shape[0]/400))) * 2,
                       (xy_candidates[i][1]+int(self.x*(frame.shape[1]/630))) * 2: (xy_candidates[i][3]+int(self.x*(frame.shape[1]/630))) * 2]

                if self.flag==1:
                    j = j + 1
                    filename = "tab/222_" + "%s.png" % str(j)
                    cv2.imwrite(filename,cand)

                result = self.find_plate(cand)
                number = ""

                if result == 0:
                    tablice.append(xy_candidates[i])
                    chars = self.Extractor.segment(cand)
                    plate_numbers = []
                    for char in chars:
                        plate_numbers.append(self.predict_character(char))

                    plate_numbers, ok = self.syntactic.check(plate_numbers)

                    for char in plate_numbers:
                        if ok == 2:

                            number = number + self.class_names[char]
                    numbers.append(number)
                    cv2.rectangle(frame, (xy_candidates[i][1]+int(self.x*(frame.shape[1]/630)), xy_candidates[i][0]+int(self.y*(frame.shape[0]/400))),
                                      (xy_candidates[i][3]+int(self.x*(frame.shape[1]/630)), xy_candidates[i][2]+int(self.y*(frame.shape[0]/400))), (0, 255, 0), 2)

            obiekt, text, rect,missing = self.tracker.update(tablice,numbers)

            szukaj = []
            miss = []
            for im in missing:
                if isinstance(im[0], list):

                    mis = self.last_frame2[(im[0][0] + int(self.y * (frame.shape[0] / 400))) * 2: (im[0][2] + int(self.y * (frame.shape[0] / 400))) * 2,(im[0][1] + int(self.x * (frame.shape[1] / 630))) * 2: (im[0][3] + int(self.x * (frame.shape[1] / 630))) * 2]


                    x0 = (im[0][1] + int(self.x * (frame.shape[1] / 630))) *2 - 20
                    x1 = (im[0][3] + int(self.x * (frame.shape[1] / 630))) * 2 + 20
                    y0 = (im[0][0] + int(self.y * (frame.shape[0] / 400))) * 2 - 10
                    y1 = (im[0][2] + int(self.y * (frame.shape[0] / 400))) * 2 + 10

                    if x0 < 0:
                        x0 = 0
                    if x1 > self.last_frame.shape[1]:
                        x1 = self.last_frame.shape[1]
                    if y0 < 0 :
                        y0 = 0
                    if y1 > self.last_frame.shape[0]:
                        y1 = self.last_frame.shape[0]
                    szukaj.append([self.last_frame[y0:y1,x0: x1],[x0,y0,x1,y1]])
                    miss.append(mis)
                else:

                    mis = self.last_frame2[(im[0] + int(self.y * (frame.shape[0] / 400))) * 2: (im[2] + int(
                        self.y * (frame.shape[0] / 400))) * 2, (im[1] + int(self.x * (frame.shape[1] / 630))) * 2: (im[
                                                                                                                        3] + int(
                        self.x * (frame.shape[1] / 630))) * 2]
                    miss.append(mis)

                    x0 = (im[1] + int(self.x * (frame.shape[1] / 630))) * 2 - 30
                    x1 = (im[3] + int(self.x * (frame.shape[1] / 630))) * 2 + 30
                    y0 = (im[0] + int(self.y * (frame.shape[0] / 400))) * 2 - 15
                    y1 = (im[2] + int(self.y * (frame.shape[0] / 400))) * 2 + 15

                    if x0 < 0:
                        x0 = 0
                    if x1 > self.last_frame.shape[1]:
                        x1 = self.last_frame.shape[1]
                    if y0 < 0 :
                        y0 = 0
                    if y1 > self.last_frame.shape[0]:
                        y1 = self.last_frame.shape[0]
                    szukaj.append([self.last_frame[y0:y1,x0: x1],[x0,y0,x1,y1]])


                #cv2.waitKey(0)
            finded = []
            for mis in range(0,len(miss)):
                FLANN_INDEX_KDITREE = 0
                MIN_MATCH_COUNT = 20
                flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
                flann = cv2.FlannBasedMatcher(flannParam, {})
                missa = cv2.cvtColor(miss[mis], cv2.COLOR_BGR2GRAY)
                szukaja = cv2.cvtColor(szukaj[mis][0], cv2.COLOR_BGR2GRAY)

                trainKP, trainDesc = sift.detectAndCompute(missa, None)

                queryKP, queryDesc = sift.detectAndCompute(szukaja, None)

                try:
                    if (type(queryDesc) != 'NoneType') or (type(trainDesc) != 'NoneType') :
                        matches = flann.knnMatch(queryDesc, trainDesc, k=2)

                        goodMatch = []
                        for m, n in matches:
                            if (m.distance < 0.75 * n.distance):
                                goodMatch.append(m)
                        if (len(goodMatch) > MIN_MATCH_COUNT):
                            tp = []
                            qp = []
                            for m in goodMatch:
                                tp.append(trainKP[m.trainIdx].pt)
                                qp.append(queryKP[m.queryIdx].pt)

                            tp, qp = np.float32((tp, qp))

                            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
                            h, w = missa.shape

                            trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])

                            queryBorder = cv2.perspectiveTransform(trainBorder, H)
                            xy = [int(queryBorder[0][0][0]), int(queryBorder[0][0][1]), int(queryBorder[0][2][0]),
                            int(queryBorder[0][2][1])]
                            tabb = [szukaj[mis][1][0]+xy[0],szukaj[mis][1][0]+xy[2],szukaj[mis][1][1]+xy[1],szukaj[mis][1][1]+xy[3]]
                            finded.append(tabb)

                        else:
                            print("Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT))
                except:
                    pass

            for find in finded:
                if find[0]<0:
                    find[0] = 0
                if find[1]<0:
                    find[1] = 0
                if find[2]<0:
                    find[2] = 0
                if find[3]<0:
                    find[3] = 0

                if find[0]>self.last_frame.shape[1]:
                    find[0] = self.last_frame.shape[1]
                if find[1]>self.last_frame.shape[1]:
                    find[1] = self.last_frame.shape[1]
                if find[2]>self.last_frame.shape[0]:
                    find[2] = self.last_frame.shape[0]
                if find[3]>self.last_frame.shape[0]:
                    find[3] = self.last_frame.shape[0]

                if find[2]>find[3]:
                    temp = find[2]
                    find[2]=find[3]
                    find[3]=temp
                if find[0]>find[1]:
                    temp = find[0]
                    find[0]=find[1]
                    find[1]=temp

                if find[2] == find[3]:
                    find[2]=0
                if find[0] == find[1]:
                    find[0]=0
                print(find[0])
                print(find[1])
                print(find[2])
                print(find[3])
                cand = self.last_frame[find[2] : find[3],find[0]: find[1] ]
                chars = self.Extractor.segment(cand)
                plate_numbers = []
                number = ""
                for char in chars:
                    plate_numbers.append(self.predict_character(char))
                plate_numbers, ok = self.syntactic.check(plate_numbers)

                for char in plate_numbers:
                    if ok == 2:

                       number = number + self.class_names[char]

                if len(number) >= 2:
                    found = 1
                    numbers.append(number)
                    tablice.append([int((find[2]-int(self.y*(frame.shape[0]/400)))/2),int((find[0]-int(self.x*(frame.shape[1]/630))) / 2),int((find[3]-int(self.y*(frame.shape[0]/400)))/2),int((find[1]-int(self.x*(frame.shape[1]/630))) / 2)])

            if found == 1:
                obiekt, text, rect, missing = self.tracker.update(tablice, numbers)

            for (objectID, centroid) in obiekt.items():

                txt = "{}".format(text.get(objectID))
                cv2.putText(frame, txt, (centroid[0]+int(self.x*(frame.shape[1]/630)) - 10, centroid[1]+int(self.y*(frame.shape[0]/400)) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape

            bytesPerLine = ch * w
            p = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = QPixmap.fromImage(p)
            pixmap = p.scaled(630, 400)

            self.pix.emit(pixmap)
            fps.update()
            self.last_frame2 = self.last_frame

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    def set_frame(self):
        ret, self.last_frame = self.video.read()
        frame = cv2.resize(self.last_frame, None, fx=0.4, fy=0.4)
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        p = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = QPixmap.fromImage(p)
        pixmap = p.scaled(630, 400)

        self.pix.emit(pixmap)

        self.video.set(cv2.CAP_PROP_POS_AVI_RATIO,0)

    def change_settings(self,localize_metgod, ekstract_method, roi):
        self.localize_method = localize_metgod

        self.exctact_method = ekstract_method

        self.roi = roi

    def apply(self):
        if self.localize_method == 1:
            self.Detector = Localize2.DetectorMorph()
        if self.localize_method == 2:
            self.Detector = Localize1.DetectorEdges()
        if self.localize_method == 3:
            self.Detector = Localize3.DetectorMorph()
        if self.exctact_method == 1:
            self.Extractor = Segmentation_chars.Segmentation()
        if self.exctact_method == 2:
            self.Extractor = Segmentation_chars2.Segmentation2()

    def set_roi(self,x,y,h,w):
        if self.roi ==1:
            self.x = x
            self.y = y
            self.height = h
            self.width = w

    def pre_process(self,frame):
        height, width, numChannels = frame.shape

        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgGrayscale = cv2.split(imgHSV)

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        imgBlurred = np.zeros((height, width, 1), np.uint8)
        imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (3, 3), 0)

        kernel_ver = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        vertical = cv2.filter2D(imgBlurred, -1, kernel_ver)

        th4 = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel_sver = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        th4 = cv2.filter2D(th4, -1, kernel_sver)

        return th4,vertical

    def filter_contour(self,threshold):
        cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        orig = threshold.copy()
        for c in cnts:
            peri = cv2.arcLength(c, True)
            if peri <= 10 or peri >= 45:
                cv2.drawContours(orig, [c], -1, (0, 255, 0), 2)

        img2 = np.zeros(orig.shape, np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)
        for i in range(1, 15):
            orig2 = orig.copy()
            kernel = np.ones((i, i), np.uint8)
            orig2 = cv2.dilate(orig2, kernel2, iterations=1)
            orig2 = cv2.morphologyEx(orig2, cv2.MORPH_CLOSE, kernel)
            orig2 = cv2.morphologyEx(orig2, cv2.MORPH_OPEN, kernel)
            nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(orig2, connectivity=8)
            for el in range(1, nlabel):
                if stats[el][2] / stats[el][3] > 3.5 and stats[el][2] / stats[el][3] < 5.5 and stats[el][3] > 10 and stats[el][3] < 40:
                    img2[labels == el] = int(255)
        return img2


    def predict_character(self,character):
        character=character.flatten()
        char = self.ocr.predict(character.reshape(1, -1))[0]
        return char


    def find_plate(self,plate):

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        logo = cv2.resize(gray, (120, 30))
        H = feature.hog(logo, orientations=9, pixels_per_cell=(6, 6),
                                    cells_per_block=(3, 3), transform_sqrt=True, block_norm="L1")
        result = self.loaded_model.predict(H.reshape(1, -1))[0]
        return result

    def save_frame(self):

        filename = "posittt/nadasie/tab15_"+"%s.bmp" % str(self.count)
        cv2.imwrite(filename, self.last_frame)
        self.count = self.count + 1

    def __del__(self):
        self.video.release()


