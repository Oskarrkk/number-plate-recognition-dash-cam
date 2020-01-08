from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import  QThread, pyqtSlot
import VideoProcess
import cv2
import numpy as np
import sys
import mainwindow


class Gui(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_CHOOSEFILE.clicked.connect(self.browse_video)
        self.pushButton_START.clicked.connect(self.play_video)
        self.pushButton_STOP.clicked.connect(self.stop_video)
        self.pushButton_CHANGE.clicked.connect(self.change)
        self.pushButton_SAVE.clicked.connect(self.save)
        self.VIDEO.mouseMoveEvent = self.mouseMoveEvent
        self.VIDEO.mousePressEvent = self.mousePressEvent
        self.VIDEO.mouseReleaseEvent = self.mouseReleaseEvent
        self.last_frame = np.zeros((1, 1))
        self.flag=0
        self.flag_browse = 0
        self.tab = 0

    def mousePressEvent(self, event):
        if self.flag_browse == 1 and self.radioButton_ROI.isChecked():
            self.originQPoint = self.VIDEO.mapFromGlobal(self.mapToGlobal(event.pos()))
            self.currentQRubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.VIDEO)
            self.currentQRubberBand.setGeometry(QtCore.QRect(self.originQPoint, QtCore.QSize()))
            self.currentQRubberBand.show()

    def mouseMoveEvent(self, event):
        if self.flag_browse == 1 and self.radioButton_ROI.isChecked():
            p = self.VIDEO.mapFromGlobal(self.mapToGlobal(event.pos()))
            if self.currentQRubberBand.isVisible() and self.VIDEO.pixmap() is not None:
                self.currentQRubberBand.setGeometry(
                    QtCore.QRect(self.originQPoint, p).normalized() & self.VIDEO.rect())

    def mouseReleaseEvent(self, event):
        if self.flag_browse == 1 and self.radioButton_ROI.isChecked():
            self.currentQRubberBand.hide()
            currentQRect = self.currentQRubberBand.geometry()
            self.Video1.set_roi(currentQRect.x(),currentQRect.y(),currentQRect.height(),currentQRect.width())
            self.currentQRubberBand.deleteLater()




    def browse_video(self):
        fdir = QtWidgets.QFileDialog.getOpenFileName(self, "Choose video file", 'C:\\Users\Oskar\Desktop\MAGISTERKA2\Projek22',"Video files (*.mp4)")
        self.flag_browse = 1
        print(fdir[0])
        if self.radioButton_Morph_2.isChecked():
            self.localize_metgod = 3
        if self.radioButton_Edge.isChecked():
            self.localize_metgod = 2
        if self.radioButton_Morph.isChecked():
            self.localize_metgod = 1
        if self.radioButton_FullImage.isChecked():
            self.roi = 2
        if self.radioButton_ROI.isChecked():
            self.roi = 1
        if self.radioButton_Segment1.isChecked():
            self.ekstract_method = 1
        if self.radioButton_Segment2.isChecked():
            self.ekstract_method = 2


        if self.flag==0:
            self.Video1 = VideoProcess.Video_Player(dir=fdir[0], method=self.localize_metgod, roi=self.roi, method2=self.ekstract_method)
            self.Video1.pix.connect(self.update_movie)
            self.Video1.frame.connect(self.detect_car)
            self.Video1.set_frame()
            print(self.Video1)


        if self.flag==1:
            self.Video1.stopped = True
            self.movie_thread.stop()
            self.Video1 = VideoProcess.Video_Player(dir=fdir[0], method=self.localize_metgod, roi=self.roi, method2=self.ekstract_method)
            self.movie_thread.Video=self.Video1
            self.Video1.pix.connect(self.update_movie)
            self.Video1.set_frame()
            self.flag = 1
            print(self.Video1)



    def play_video(self):

        if self.flag == 1:
            self.Video1.stopped = False
            self.movie_thread.start()
            self.flag = 1
            print(self.Video1)

        if self.flag == 0:
            self.Video1.stopped = False
            self.movie_thread = MovieThread(self.Video1)
            self.movie_thread.start()
            self.flag = 1
            print(self.movie_thread)


    def stop_video(self):

        if self.flag == 1:
            self.Video1.stopped = True
            self.flag = 1
            self.Video1.video.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
            self.movie_thread.stop()
            print(self.movie_thread)


    def save(self):
        self.Video1.save_frame()


    def change(self):
        if self.radioButton_Morph_2.isChecked():
            self.localize_metgod = 3
        if self.radioButton_Edge.isChecked():
            self.localize_metgod = 2
        if self.radioButton_Morph.isChecked():
            self.localize_metgod = 1
        if self.radioButton_FullImage.isChecked():
            self.roi = 2
        if self.radioButton_ROI.isChecked():
            self.roi = 1
        if self.radioButton_Segment1.isChecked():
            self.ekstract_method = 1
        if self.radioButton_Segment2.isChecked():
            self.ekstract_method = 2
        self.Video1.change_settings(self.localize_metgod,self.ekstract_method,self.roi)





    @pyqtSlot(QPixmap)
    def update_movie(self,pixmap):
        self.VIDEO.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def detect_car(self,frame):
        cv2.imshow("crop", frame)


    @pyqtSlot(np.ndarray)
    def detect_plate(self,numers):
        pass


#Processing video in different Thread
class MovieThread(QThread):
    def __init__(self, Video):
        super().__init__()
        QThread.__init__(self)
        self.Video = Video
        self.num_frames = int(self.Video.video.get(cv2.CAP_PROP_FRAME_COUNT))
    def run(self):

        self.Video.get_frame()


    def stop(self):
        self.wait()




def main():
    app = QtWidgets.QApplication(sys.argv)
    form = Gui()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()