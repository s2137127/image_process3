from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from main_ui import Ui_MainWindow
import cv2
import numpy as np
import re
import pyqtgraph as pg


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
        # print("path: ", self.img_path)
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

    def process_Clicked(self):
        self.label_2.setText("wait..")
        size = int(self.filter_size.text().split(' ')[-1])
        s = int((size - 1) / 2)
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img, np.uint8)

        filter = np.zeros((size, size))
        tmp_arr = re.split('\n|,', self.plainTextEdit_filter.toPlainText())
        if len(tmp_arr) == size ** 2:
            for i in range(size):
                for j in range(size):
                    filter[i, j] = tmp_arr[j + i * size]

            for i in range(s, origin.shape[0] - s):
                for j in range(s, origin.shape[1] - s):
                    new[i - s, j - s, 0] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 0], filter)))))))
                    new[i - s, j - s, 1] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 1], filter)))))))
                    new[i - s, j - s, 2] = np.min((255, np.max(
                        (0, int(np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 2], filter)))))))

            self.qimg = QImage(new, new.shape[1], new.shape[0], 3 * new.shape[1], QImage.Format_RGB888).rgbSwapped()
            self.process_label.setPixmap(QPixmap.fromImage(self.qimg))
            self.process_label.setScaledContents(True)

        else:
            self.plainTextEdit_filter.setText("size不符合你的輸入矩陣")
        self.label_2.setText("finish..")

    def local_enhance_Clicked(self):

        filter_size = self.local_fsize.value()
        print("waiting")
        self.label.clear()
        k0 = self.doubleSpinBox_k0.value()
        k1 = self.doubleSpinBox_k1.value()
        k2 = self.doubleSpinBox_k2.value()
        k3 = self.doubleSpinBox_k3.value()
        C = self.doubleSpinBox_C.value()
        s = int((filter_size - 1) / 2)
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img, np.uint8)
        filter = np.ones((filter_size, filter_size))
        Mg = [np.mean(self.img[:, :, 0]), np.mean(self.img[:, :, 1]), np.mean(self.img[:, :, 2])]
        Sg = [np.std(self.img[:, :, 0]), np.std(self.img[:, :, 1]), np.std(self.img[:, :, 2])]
        for c in range(origin.shape[2]):
            for i in range(s, origin.shape[0] - s):
                for j in range(s, origin.shape[1] - s):
                    tmp = np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, c], filter)
                    Ml = np.mean(tmp)
                    Sl = np.std(tmp)
                    # print(j)
                    if k0 * Mg[c] < Ml and k1 * Mg[c] > Ml and k2 * Sg[c] < Sl and k3 * Sg[c] > Sl:
                        new[i - s, j - s, c] = C * self.img[i - s, j - s, c]
                    else:
                        new[i - s, j - s, c] = self.img[i - s, j - s, c]
        self.qimg = QImage(new, new.shape[1], new.shape[0], 3 * new.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.local_enhance_label.setPixmap(QPixmap.fromImage(self.qimg))
        self.local_enhance_label.setScaledContents(True)
        self.label.setText("finish..")
        # print(new.shape,self.img.shape)
        print("finish")

    def equalize_Clicked(self):
        self.histogram = np.zeros(256)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                self.histogram[self.img[i, j, 0]] += 1
        prob = self.histogram / (self.img.shape[0] * self.img.shape[1])
        s = np.zeros(256)
        # histogram_eq = np.zeros(256)
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

    def pushButton_LOG_Clicked(self):
        fsize = self.fsize_log.value()
        gaussian = np.zeros((fsize, fsize))
        sigma = self.log_sigma.value()
        thres = self.log_thres.value()
        lapla = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        s = int((fsize - 1) / 2)
        origin = np.zeros((self.gray.shape[0] + 2 * s, self.gray.shape[1] + 2 * s, 3))
        origin[s:-s, s:-s] = self.gray.copy()

        new_log = np.zeros_like(self.gray, float)
        new_zero = np.zeros_like(self.gray, np.uint8)
        # gaussian
        for i in range(fsize):
            for j in range(fsize):
                # filter_log[i, j] = np.exp(-(j ** 2 + i ** 2) / (2 * sigma ** 2)) * (
                #             j ** 2 + i ** 2 - 2 * sigma ** 2) / (sigma ** 4)
                gaussian[i, j] = np.exp(-(j ** 2 + i ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

        for i in range(s, origin.shape[0] - s):
            for j in range(s, origin.shape[1] - s):
                new_log[i - s, j - s, :] = np.sum(np.multiply(origin[i - s:i + s + 1, j - s:j + s + 1, 0], gaussian))

        new_log_pad = np.zeros((new_log.shape[0] + 2, new_log.shape[1] + 2, 3),float)
        new_log2 = np.zeros_like(new_log,float)
        # print(new_log.shape)
        new_log_pad[1:-1, 1:-1, :] = new_log.copy()
        new_log2_show = np.zeros_like(new_log2,np.uint8)

        # laplacian
        for i in range(1,new_log_pad.shape[0]-1):
            for j in range(1,new_log_pad.shape[1]-1):
                new_log2[i - 1, j - 1, :] = np.sum(np.multiply(new_log_pad[i - 1:i + 2, j - 1:j + 2, 0], lapla))
        for i in range(new_log2_show.shape[0]):
            for j in range(new_log2_show.shape[1]):
                if new_log2[i,j,0]>255:
                    new_log2_show[i,j,:] = 255
                elif new_log2[i,j,0]<0:
                    new_log2_show[i,j,:]=0
                else:
                    new_log2_show[i,j,:] = new_log2[i,j,:]*(255 / np.max(new_log2))
        print(new_log2_show,np.max(new_log2_show),new_log2_show.dtype)
        # new_log2 = np.array(new_log.copy(),np.uint8)
        new_log2_pad = np.zeros((self.gray.shape[0] + 2, self.gray.shape[1] + 2, 3))
        new_log2_pad[1:-1, 1:-1, :] = new_log2.copy()
        for i in range(1, new_log2_pad.shape[0] - 1):
            for j in range(1, new_log2_pad.shape[1] - 1):
                tmp = np.array(new_log2_pad[i - 1:i + 2, j - 1:j + 2 , 0], int)

                # print(tmp)
                if (tmp[0, 0] * tmp[2, 2] < 0 and abs(tmp[0, 0] - tmp[2, 2]) > thres) or \
                        (tmp[0, 2] * tmp[2, 0] < 0 and abs(tmp[0, 2] - tmp[2, 0]) > thres) or \
                        (tmp[0, 0] * tmp[2, 0] and abs(tmp[0, 0] - tmp[2, 0]) > thres) < 0 or \
                        (tmp[2, 0] * tmp[2, 2] < 0 and abs(tmp[2, 0] - tmp[2, 2]) > thres) or \
                        (tmp[0, 0] * tmp[0, 2] and abs(tmp[0, 0] - tmp[0, 2]) > thres) < 0 or \
                        (tmp[0, 2] * tmp[2, 2] < 0 and abs(tmp[0, 2] - tmp[2, 2]) > thres) or \
                        (tmp[1, 0] * tmp[1, 2] < 0 and abs(tmp[1, 0] - tmp[1, 2]) > thres) or \
                        (tmp[0, 1] * tmp[2, 1] < 0 and abs(tmp[0, 1] - tmp[2, 1]) > thres):
                    new_zero[i - 1, j - 1, :] = 255

        qimg = QImage(new_zero.data, new_zero.shape[1], new_zero.shape[0], 3 * new_zero.shape[1],
                      QImage.Format_RGB888).rgbSwapped()
        qimg2 = QImage(new_log2_show.data, new_log2_show.shape[1], new_log2_show.shape[0], 3 * new_log2_show.shape[1],
                       QImage.Format_RGB888).rgbSwapped()
        self.label_zero_cross.setPixmap(QPixmap.fromImage(qimg))
        self.label_zero_cross.setScaledContents(True)
        self.label_LOG.setPixmap(QPixmap.fromImage(qimg2))
        self.label_LOG.setScaledContents(True)
        print('finish')


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
