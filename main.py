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
        self.setupUi(self)
        self.img_path = None
        self.img = None
        self.select_pic.clicked.connect(self.select_pic_Clicked)
        self.process.clicked.connect(self.process_Clicked)
        self.local_enhance.clicked.connect(self.local_enhance_Clicked)
    def select_pic_Clicked(self):
        # 開啟資料夾選則照片
        self.img_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open file",
                                                       "./",
                                                       "Images (*.png *.BMP *.jpg)")
        # print("path: ", self.img_path)
        self.img = cv2.imread(self.img_path)  # 讀檔

        height, width, channel = self.img.shape
        self.qimg = QImage(self.img, width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.origin_label.setPixmap(QPixmap.fromImage(self.qimg))
        self.origin_label.setScaledContents(True)

    def process_Clicked(self):
        self.label_2.setText("wait..")
        size = int(self.filter_size.text().split(' ')[-1])
        s = int((size - 1) / 2)
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img, np.uint8)

        filter = np.zeros((size, size))
        tmp_arr = re.split('\n|,',self.plainTextEdit_filter.toPlainText())
        if len(tmp_arr) == size ** 2:
            for i in range(size):
                for j in range(size):
                    filter[i, j] = tmp_arr[j + i * size]

            for i in range(s, origin.shape[0] - s-int((size-1)/2)):
                for j in range(s, origin.shape[1] - s-int((size-1)/2)):
                    a = int(np.sum(np.multiply(origin[i:i + size, j:j + size, 0], filter)))

                    new[i - s, j - s, 0] = np.min((255,np.max((0,int(np.sum(np.multiply(origin[i:i + size, j:j + size, 0], filter)))))))
                    new[i - s, j - s, 1] = np.min((255,np.max((0,int(np.sum(np.multiply(origin[i:i + size, j:j + size, 1], filter)))))))
                    new[i - s, j - s, 2] = np.min((255,np.max((0,int(np.sum(np.multiply(origin[i:i + size, j:j + size, 2], filter)))))))

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
        C= self.doubleSpinBox_C.value()
        s = int((filter_size-1)/2)
        origin = np.zeros((self.img.shape[0] + 2 * s, self.img.shape[1] + 2 * s, 3), np.uint8)
        origin[s:-s, s:-s] = self.img.copy()
        new = np.zeros_like(self.img,np.uint8)
        filter = np.ones((filter_size,filter_size))
        Mg = [np.mean(self.img[:,:,0]),np.mean(self.img[:,:,1]),np.mean(self.img[:,:,2])]
        Sg = [np.std(self.img[:,:,0]),np.std(self.img[:,:,1]),np.std(self.img[:,:,2])]
        for c in range(origin.shape[2]):
            for i in range(s, origin.shape[0] - s-int((filter_size-1)/2)):
                for j in range(s, origin.shape[0] - s-int((filter_size-1)/2)):
                    tmp = np.multiply(origin[i:i + filter_size, j:j + filter_size, c], filter)
                    Ml = np.mean(tmp)
                    Sl = np.std(tmp)
                    if k0*Mg[c]<Ml and k1*Mg[c]>Ml and k2*Sg[c] < Sl and k3*Sg[c]>Sl:
                        new[i-s,j-s,c] = C* self.img[i-s,j-s,c]
                    else:
                        new[i - s, j - s, c] = self.img[i - s, j - s, c]
        self.qimg = QImage(new, new.shape[1], new.shape[0], 3 * new.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.local_enhance_label.setPixmap(QPixmap.fromImage(self.qimg))
        self.local_enhance_label.setScaledContents(True)
        self.label.setText("finish..")
        print("finish")
def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
