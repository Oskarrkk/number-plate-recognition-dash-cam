# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(996, 547)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_CHOOSEFILE = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_CHOOSEFILE.setGeometry(QtCore.QRect(670, 20, 91, 41))
        self.pushButton_CHOOSEFILE.setObjectName("pushButton_CHOOSEFILE")
        self.pushButton_START = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_START.setGeometry(QtCore.QRect(40, 440, 81, 51))
        self.pushButton_START.setObjectName("pushButton_START")
        self.pushButton_STOP = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_STOP.setGeometry(QtCore.QRect(160, 440, 81, 51))
        self.pushButton_STOP.setObjectName("pushButton_STOP")
        self.pushButton_SAVE = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_SAVE.setGeometry(QtCore.QRect(810, 20, 81, 41))
        self.pushButton_SAVE.setObjectName("pushButton_SAVE")
        self.VIDEO = QtWidgets.QLabel(self.centralwidget)
        self.VIDEO.setGeometry(QtCore.QRect(10, 10, 641, 401))
        self.VIDEO.setText("")
        self.VIDEO.setObjectName("VIDEO")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(660, 70, 291, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(680, 110, 291, 51))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton_Morph_2 = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_Morph_2.setObjectName("radioButton_Morph_2")
        self.horizontalLayout.addWidget(self.radioButton_Morph_2)
        self.radioButton_Edge = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_Edge.setObjectName("radioButton_Edge")
        self.horizontalLayout.addWidget(self.radioButton_Edge)
        self.radioButton_Morph = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_Morph.setObjectName("radioButton_Morph")
        self.radioButton_Morph_2.setChecked(1)
        self.horizontalLayout.addWidget(self.radioButton_Morph)
        self.layoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(680, 290, 221, 51))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_FullImage = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioButton_FullImage.setObjectName("radioButton_FullImage")
        self.radioButton_FullImage.setChecked(1)
        self.horizontalLayout_2.addWidget(self.radioButton_FullImage)
        self.radioButton_ROI = QtWidgets.QRadioButton(self.layoutWidget_2)
        self.radioButton_ROI.setObjectName("radioButton_ROI")
        self.horizontalLayout_2.addWidget(self.radioButton_ROI)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(680, 90, 241, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(680, 270, 241, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_PAUSE = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_PAUSE.setGeometry(QtCore.QRect(290, 440, 81, 51))
        self.pushButton_PAUSE.setObjectName("pushButton_PAUSE")
        self.pushButton_CHANGE = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_CHANGE.setGeometry(QtCore.QRect(680, 360, 121, 51))
        self.pushButton_CHANGE.setObjectName("pushButton_CHANGE")
        self.layoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_3.setGeometry(QtCore.QRect(680, 200, 221, 51))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.radioButton_Segment1 = QtWidgets.QRadioButton(self.layoutWidget_3)
        self.radioButton_Segment1.setObjectName("radioButton_Segment1")
        self.radioButton_Segment1.setChecked(1)
        self.horizontalLayout_3.addWidget(self.radioButton_Segment1)
        self.radioButton_Segment2 = QtWidgets.QRadioButton(self.layoutWidget_3)
        self.radioButton_Segment2.setObjectName("radioButton_Segment2")
        self.horizontalLayout_3.addWidget(self.radioButton_Segment2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(680, 180, 241, 16))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_CHOOSEFILE.setText(_translate("MainWindow", "LOAD VIDEO"))
        self.pushButton_START.setText(_translate("MainWindow", "START"))
        self.pushButton_STOP.setText(_translate("MainWindow", "STOP"))
        self.pushButton_SAVE.setText(_translate("MainWindow", "Save_frame"))
        self.radioButton_Morph_2.setText(_translate("MainWindow", "Morphological"))
        self.radioButton_Edge.setText(_translate("MainWindow", "Edge"))
        self.radioButton_Morph.setText(_translate("MainWindow", "Morphological + Edge"))
        self.radioButton_FullImage.setText(_translate("MainWindow", "Full Video"))
        self.radioButton_ROI.setText(_translate("MainWindow", "Other"))
        self.label.setText(_translate("MainWindow", "SELECT METHOD OF NUMBER PLATE DETECTION"))
        self.label_2.setText(_translate("MainWindow", "SELECT ROI:"))
        self.pushButton_PAUSE.setText(_translate("MainWindow", "PAUSE"))
        self.pushButton_CHANGE.setText(_translate("MainWindow", "CHANGE"))
        self.radioButton_Segment1.setText(_translate("MainWindow", "Segmentation 1"))
        self.radioButton_Segment2.setText(_translate("MainWindow", "Segmentation 2"))
        self.label_3.setText(_translate("MainWindow", "SELECT METHOD OF SEGMENTATION"))
