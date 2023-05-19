from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import re
import pyqtgraph as pg
import time
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1126, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 171, 171))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.origin_label = QtWidgets.QLabel(self.layoutWidget)
        self.origin_label.setObjectName("origin_label")
        self.verticalLayout.addWidget(self.origin_label)
        self.select_pic = QtWidgets.QPushButton(self.layoutWidget)
        self.select_pic.setObjectName("select_pic")
        self.verticalLayout.addWidget(self.select_pic)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(230, 21, 311, 171))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.process_label = QtWidgets.QLabel(self.layoutWidget1)
        self.process_label.setObjectName("process_label")
        self.horizontalLayout.addWidget(self.process_label)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.filter_size = QtWidgets.QSpinBox(self.layoutWidget1)
        self.filter_size.setMinimum(2)
        self.filter_size.setMaximum(7)
        self.filter_size.setSingleStep(2)
        self.filter_size.setProperty("value", 3)
        self.filter_size.setObjectName("filter_size")
        self.verticalLayout_2.addWidget(self.filter_size)
        self.plainTextEdit_filter = QtWidgets.QPlainTextEdit(self.layoutWidget1)
        self.plainTextEdit_filter.setOverwriteMode(True)
        self.plainTextEdit_filter.setObjectName("plainTextEdit_filter")
        self.verticalLayout_2.addWidget(self.plainTextEdit_filter)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.process = QtWidgets.QPushButton(self.layoutWidget1)
        self.process.setObjectName("process")
        self.verticalLayout_2.addWidget(self.process)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 11)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 7)
        self.horizontalLayout.setStretch(1, 6)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(20, 220, 268, 201))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_2.addWidget(self.lineEdit_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_2.addWidget(self.lineEdit_4)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_2.addWidget(self.lineEdit_5)
        self.horizontalLayout_2.setStretch(0, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.doubleSpinBox_k0 = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_k0.setSingleStep(0.1)
        self.doubleSpinBox_k0.setObjectName("doubleSpinBox_k0")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_k0)
        self.doubleSpinBox_k1 = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_k1.setSingleStep(0.1)
        self.doubleSpinBox_k1.setProperty("value", 0.1)
        self.doubleSpinBox_k1.setObjectName("doubleSpinBox_k1")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_k1)
        self.doubleSpinBox_k2 = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_k2.setSingleStep(0.1)
        self.doubleSpinBox_k2.setObjectName("doubleSpinBox_k2")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_k2)
        self.doubleSpinBox_k3 = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_k3.setSingleStep(0.1)
        self.doubleSpinBox_k3.setProperty("value", 0.1)
        self.doubleSpinBox_k3.setObjectName("doubleSpinBox_k3")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_k3)
        self.doubleSpinBox_C = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_C.setSingleStep(0.1)
        self.doubleSpinBox_C.setProperty("value", 22.8)
        self.doubleSpinBox_C.setObjectName("doubleSpinBox_C")
        self.horizontalLayout_5.addWidget(self.doubleSpinBox_C)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.local_enhance_label = QtWidgets.QLabel(self.layoutWidget2)
        self.local_enhance_label.setObjectName("local_enhance_label")
        self.horizontalLayout_6.addWidget(self.local_enhance_label)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.local_enhance = QtWidgets.QPushButton(self.layoutWidget2)
        self.local_enhance.setObjectName("local_enhance")
        self.verticalLayout_4.addWidget(self.local_enhance)
        self.local_fsize = QtWidgets.QSpinBox(self.layoutWidget2)
        self.local_fsize.setSingleStep(2)
        self.local_fsize.setProperty("value", 3)
        self.local_fsize.setObjectName("local_fsize")
        self.verticalLayout_4.addWidget(self.local_fsize)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)
        self.horizontalLayout_6.setStretch(0, 7)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 7)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(340, 240, 231, 221))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.equalize = QtWidgets.QPushButton(self.layoutWidget3)
        self.equalize.setObjectName("equalize")
        self.verticalLayout_5.addWidget(self.equalize)
        self.equalize_label = QtWidgets.QLabel(self.layoutWidget3)
        self.equalize_label.setObjectName("equalize_label")
        self.verticalLayout_5.addWidget(self.equalize_label)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(620, 70, 381, 171))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.verticalLayout_11.addWidget(self.lineEdit_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalSlider = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setProperty("value", 100)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_7.addWidget(self.horizontalSlider)
        self.log_thres = QtWidgets.QSpinBox(self.widget)
        self.log_thres.setMaximum(255)
        self.log_thres.setProperty("value", 100)
        self.log_thres.setObjectName("log_thres")
        self.horizontalLayout_7.addWidget(self.log_thres)
        self.verticalLayout_11.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8.addLayout(self.verticalLayout_11)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.verticalLayout_10.addWidget(self.lineEdit_8)
        self.log_sigma = QtWidgets.QDoubleSpinBox(self.widget)
        self.log_sigma.setProperty("value", 4.0)
        self.log_sigma.setObjectName("log_sigma")
        self.verticalLayout_10.addWidget(self.log_sigma)
        self.horizontalLayout_8.addLayout(self.verticalLayout_10)
        self.verticalLayout_9.addLayout(self.horizontalLayout_8)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(1, 1, 1, 2)
        self.verticalLayout_7.setSpacing(1)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.verticalLayout_7.addWidget(self.lineEdit_7)
        self.fsize_log = QtWidgets.QSpinBox(self.widget)
        self.fsize_log.setSingleStep(2)
        self.fsize_log.setProperty("value", 25)
        self.fsize_log.setObjectName("fsize_log")
        self.verticalLayout_7.addWidget(self.fsize_log)
        self.pushButton_LOG = QtWidgets.QPushButton(self.widget)
        self.pushButton_LOG.setObjectName("pushButton_LOG")
        self.verticalLayout_7.addWidget(self.pushButton_LOG)
        self.verticalLayout_9.addLayout(self.verticalLayout_7)
        self.horizontalLayout_4.addLayout(self.verticalLayout_9)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(582, 245, 531, 191))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_LOG = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_LOG.sizePolicy().hasHeightForWidth())
        self.label_LOG.setSizePolicy(sizePolicy)
        self.label_LOG.setObjectName("label_LOG")
        self.horizontalLayout_3.addWidget(self.label_LOG)
        self.label_zero_cross = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_zero_cross.sizePolicy().hasHeightForWidth())
        self.label_zero_cross.setSizePolicy(sizePolicy)
        self.label_zero_cross.setObjectName("label_zero_cross")
        self.horizontalLayout_3.addWidget(self.label_zero_cross)
        self.label_sobel = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sobel.sizePolicy().hasHeightForWidth())
        self.label_sobel.setSizePolicy(sizePolicy)
        self.label_sobel.setObjectName("label_sobel")
        self.horizontalLayout_3.addWidget(self.label_sobel)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1126, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.horizontalSlider.valueChanged['int'].connect(self.log_thres.setValue) # type: ignore
        self.log_thres.valueChanged['int'].connect(self.horizontalSlider.setValue) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.origin_label.setText(_translate("MainWindow", "origin"))
        self.select_pic.setText(_translate("MainWindow", "select picture"))
        self.process_label.setText(_translate("MainWindow", "TextLabel"))
        self.filter_size.setPrefix(_translate("MainWindow", "size: "))
        self.plainTextEdit_filter.setPlainText(_translate("MainWindow", "請用逗號間格數字\n"
""))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.process.setText(_translate("MainWindow", "process"))
        self.lineEdit.setText(_translate("MainWindow", "k0"))
        self.lineEdit_3.setText(_translate("MainWindow", "k1"))
        self.lineEdit_2.setText(_translate("MainWindow", "k2"))
        self.lineEdit_4.setText(_translate("MainWindow", "k3"))
        self.lineEdit_5.setText(_translate("MainWindow", "C"))
        self.local_enhance_label.setText(_translate("MainWindow", "TextLabel"))
        self.local_enhance.setText(_translate("MainWindow", "local enhance"))
        self.local_fsize.setPrefix(_translate("MainWindow", "filter size "))
        self.equalize.setText(_translate("MainWindow", "equalize"))
        self.equalize_label.setText(_translate("MainWindow", "TextLabel"))
        self.lineEdit_6.setText(_translate("MainWindow", "threshold"))
        self.lineEdit_8.setText(_translate("MainWindow", "sigma of LOG"))
        self.lineEdit_7.setText(_translate("MainWindow", "filter_size"))
        self.pushButton_LOG.setText(_translate("MainWindow", "show all"))
        self.label_LOG.setText(_translate("MainWindow", "LOG"))
        self.label_zero_cross.setText(_translate("MainWindow", "zero-cross"))
        self.label_sobel.setText(_translate("MainWindow", "sobel"))





class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.gray = None
        self.setupUi(self)
        self.img_path = None
        self.histogram = []
        self.img = None
        self.select_pic.clicked.connect(self.select_pic_Clicked)
        self.process.clicked.connect(self.process_Clicked)
        self.local_enhance.clicked.connect(self.local_enhance_Clicked)
        self.equalize.clicked.connect(self.equalize_Clicked)
        self.pushButton_LOG.clicked.connect(self.pushButton_LOG_Clicked)

    def select_pic_Clicked(self):
        # 開啟資料夾選則照片
        self.img_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open file",
                                                       "./",
                                                       "Images (*.png *.BMP *.jpg)")

        self.img = cv2.imread(self.img_path)  # 讀檔

        height, width, channel = self.img.shape
        qimg = QImage(self.img, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.origin_label.setPixmap(QPixmap.fromImage(qimg))
        self.origin_label.setScaledContents(True)
        # 取灰階圖
        self.gray = np.zeros_like(self.img, np.uint8)
        for i in range(self.gray.shape[0]):
            for j in range(self.gray.shape[1]):
                self.gray[i, j, :] = np.mean(self.img[i, j, :])

    def process_Clicked(self):#part2:可以自己選擇filter

        start = time.time()#開始時間
        size = int(self.filter_size.text().split(' ')[-1])
        s = int((size - 1) / 2)
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img, np.uint8)

        filter = np.zeros((size, size))
        tmp_arr = re.split('\n|,', self.plainTextEdit_filter.toPlainText())#將使用者輸入的filter轉成list個別存起來
        print(tmp_arr)
        print(len(tmp_arr),size**2)
        if len(tmp_arr) == size ** 2:
            for i in range(size):#創建使用者輸入的filter
                for j in range(size):

                    filter[i, j] = float(tmp_arr[j + i * size])

            for i in range(s, origin.shape[0] - s):#用filter對圖片做convolution
                for j in range(s, origin.shape[1] - s):
                    new[i - s, j - s, 0] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 0], filter)))))))
                    new[i - s, j - s, 1] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 1], filter)))))))
                    new[i - s, j - s, 2] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 2], filter)))))))

            self.qimg = QImage(new, new.shape[1], new.shape[0], 3 * new.shape[1], QImage.Format_RGB888).rgbSwapped()#set & show image
            self.process_label.setPixmap(QPixmap.fromImage(self.qimg))
            self.process_label.setScaledContents(True)

        else:#filter dismatch the input size value
            self.plainTextEdit_filter.setPlainText("size不符合你的輸入矩陣或輸入有誤")
        self.label_2.setText("finish..")
        end = time.time()#end time
        self.label_2.setText('運算時間:\n %f' %(end-start))
    def local_enhance_Clicked(self):#part 4

        filter_size = self.local_fsize.value()#read filter size
        print("waiting")
        k0 = self.doubleSpinBox_k0.value()#read parameter of k0,k1,k2,k3,C
        k1 = self.doubleSpinBox_k1.value()
        k2 = self.doubleSpinBox_k2.value()
        k3 = self.doubleSpinBox_k3.value()
        C = self.doubleSpinBox_C.value()
        s = int((filter_size - 1) / 2)
        #padding origin img
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img, np.uint8)
        filter = np.ones((filter_size, filter_size))
        #save global mean&std
        Mg = [np.mean(self.img[:, :, 0]), np.mean(self.img[:, :, 1]), np.mean(self.img[:, :, 2])]
        Sg = [np.std(self.img[:, :, 0]), np.std(self.img[:, :, 1]), np.std(self.img[:, :, 2])]
        #local enhance
        for c in range(origin.shape[2]):
            for i in range(s, origin.shape[0] - s):
                for j in range(s, origin.shape[1] - s):
                    tmp = np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, c], filter)
                    Ml = np.mean(tmp)
                    Sl = np.std(tmp)

                    if k0 * Mg[c] < Ml and k1 * Mg[c] > Ml and k2 * Sg[c] < Sl and k3 * Sg[c] > Sl:
                        new[i - s, j - s, c] = C * self.img[i - s, j - s, c]
                    else:
                        new[i - s, j - s, c] = self.img[i - s, j - s, c]
        self.qimg = QImage(new, new.shape[1], new.shape[0], 3 * new.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.local_enhance_label.setPixmap(QPixmap.fromImage(self.qimg))
        self.local_enhance_label.setScaledContents(True)

        print("finish")

    def equalize_Clicked(self):#equalize
        #make histogram
        self.histogram = np.zeros(256)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                self.histogram[self.img[i, j, 0]] += 1
        #make probability
        prob = self.histogram / (self.img.shape[0] * self.img.shape[1])
        s = np.zeros(256)

        new = np.zeros_like(self.img, np.uint8)
        for i in range(256):  # 製作各pixel value對應的s
            if i > 1:
                s[i] = s[i - 1] + prob[i]
            else:
                s[i] = prob[i]
        s *= 255

        for i in range(self.img.shape[0]):  # 轉換 pixel value
            for j in range(self.img.shape[1]):
                for c in range(self.img.shape[2]):
                    new[i, j, c] = s[self.img[i, j, c]]
                # histogram_eq[new[i, j, 0]] += 1
        qimg = QImage(new.data, new.shape[1], new.shape[0], 3 * new.shape[1],
                      QImage.Format_RGB888).rgbSwapped()
        self.equalize_label.setPixmap(QPixmap.fromImage(qimg))
        self.equalize_label.setScaledContents(True)
    def func_sobel(self):#sobel
        origin = np.zeros((self.gray.shape[0]+2,self.gray.shape[1]+2,3))
        origin[1:-1, 1:-1] = self.gray.copy()
        new = np.zeros_like(self.gray,np.uint8)
        sobel_h = np.array(([1,0,-1],[2,0,-2],[1,0,-1]))#horizontal sobel
        sobel_v = np.array(([1,2,1],[0,0,0],[-1,-2,-1]))#vertical sobel
        #convolution img & sobel
        for i in range(1, origin.shape[0] - 1):
            for j in range(1, origin.shape[1] - 1):
                a =  int(np.sum(np.multiply(origin[i - 1:i + 2, j - 1:j + 2, 0], sobel_h))**2)
                b =  int(np.sum(np.multiply(origin[i - 1:i + 2, j - 1:j + 2, 0], sobel_v))**2)
                new[i - 1, j - 1, :] = (a+b)**0.5
        return new

    def pushButton_LOG_Clicked(self):#LoG
        print("start...")
        fsize = self.fsize_log.value()
        gaussian = np.zeros((fsize, fsize))
        sigma = self.log_sigma.value()
        thres = self.log_thres.value()
        lapla = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])#laplacian filter
        s = int((fsize - 1) / 2)
        origin = np.zeros((self.gray.shape[0] + 2 * s, self.gray.shape[1] + 2 * s, 3))
        origin[s:-s, s:-s] = self.gray.copy()

        new_log = np.zeros_like(self.gray, float)
        new_zero = np.zeros_like(self.gray, np.uint8)
        # make gaussian filter
        for i in range(fsize):
            for j in range(fsize):
                # filter_log[i, j] = np.exp(-(j ** 2 + i ** 2) / (2 * sigma ** 2)) * (
                #             j ** 2 + i ** 2 - 2 * sigma ** 2) / (sigma ** 4)
                gaussian[i, j] = np.exp(-(j ** 2 + i ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        #conv with gaussion filter
        for i in range(s, origin.shape[0] - s):
            for j in range(s, origin.shape[1] - s):
                new_log[i - s, j - s, :] = np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 0], gaussian))

        new_log_pad = np.zeros((new_log.shape[0] + 2, new_log.shape[1] + 2, 3),float)
        new_log2 = np.zeros_like(new_log,float)
        # padding
        new_log_pad[1:-1, 1:-1, :] = new_log.copy()
        new_log2_show = np.zeros_like(new_log2,np.uint8)

        # conv with laplacian
        for i in range(1,new_log_pad.shape[0]-1):
            for j in range(1,new_log_pad.shape[1]-1):
                new_log2[i - 1, j - 1, :] = np.sum(np.multiply(new_log_pad[i - 1:i + 2, j - 1:j + 2, 0], lapla))
        #clipping LoG img to [0,255] for  exhibit LoG
        for i in range(new_log2_show.shape[0]):
            for j in range(new_log2_show.shape[1]):
                if new_log2[i,j,0]>255:
                    new_log2_show[i,j,:] = 255
                elif new_log2[i,j,0]<0:
                    new_log2_show[i,j,:]=0
                else:
                    new_log2_show[i,j,:] = new_log2[i,j,:]*(255 / np.max(new_log2))
        #padding
        new_log2_pad = np.zeros((self.gray.shape[0] + 2, self.gray.shape[1] + 2, 3))
        new_log2_pad[1:-1, 1:-1, :] = new_log2.copy()
        #zero-crossing
        for i in range(1, new_log2_pad.shape[0] - 1):
            for j in range(1, new_log2_pad.shape[1] - 1):
                tmp = np.array(new_log2_pad[i - 1:i + 2, j - 1:j + 2 , 0], int)

                if (tmp[0, 0] * tmp[2, 2] < 0 and abs(tmp[0, 0] - tmp[2, 2]) > thres) or \
                        (tmp[0, 2] * tmp[2, 0] < 0 and abs(tmp[0, 2] - tmp[2, 0]) > thres) or \
                        (tmp[0, 0] * tmp[2, 0] and abs(tmp[0, 0] - tmp[2, 0]) > thres) < 0 or \
                        (tmp[2, 0] * tmp[2, 2] < 0 and abs(tmp[2, 0] - tmp[2, 2]) > thres) or \
                        (tmp[0, 0] * tmp[0, 2] and abs(tmp[0, 0] - tmp[0, 2]) > thres) < 0 or \
                        (tmp[0, 2] * tmp[2, 2] < 0 and abs(tmp[0, 2] - tmp[2, 2]) > thres) or \
                        (tmp[1, 0] * tmp[1, 2] < 0 and abs(tmp[1, 0] - tmp[1, 2]) > thres) or \
                        (tmp[0, 1] * tmp[2, 1] < 0 and abs(tmp[0, 1] - tmp[2, 1]) > thres):
                    new_zero[i - 1, j - 1, :] = 255
        #making sobel img
        sobel_img = self.func_sobel()

        qimg = QImage(new_zero.data, new_zero.shape[1], new_zero.shape[0], 3 * new_zero.shape[1],
                      QImage.Format_RGB888).rgbSwapped()
        qimg2 = QImage(new_log2_show.data, new_log2_show.shape[1], new_log2_show.shape[0], 3 * new_log2_show.shape[1],
                       QImage.Format_RGB888).rgbSwapped()

        qimg3 = QImage(sobel_img.data, sobel_img.shape[1], sobel_img.shape[0], 3 * sobel_img.shape[1],
                       QImage.Format_RGB888).rgbSwapped()

        self.label_zero_cross.setPixmap(QPixmap.fromImage(qimg))
        self.label_zero_cross.setScaledContents(True)
        self.label_LOG.setPixmap(QPixmap.fromImage(qimg2))
        self.label_LOG.setScaledContents(True)
        self.label_sobel.setPixmap(QPixmap.fromImage(qimg3))
        self.label_sobel.setScaledContents(True)
        print('finish')


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
