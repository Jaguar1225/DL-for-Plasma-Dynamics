
import sys

from KSP_connect import *
from Clustering import *
from NNPred import *

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt,QPointF, QTimer
from PyQt5 import uic, QtGui
from PyQt5.QtChart import QChart, QLineSeries, QValueAxis

x = np.linspace(0,100,100)
y = np.sin(x)

form_class = uic.loadUiType("Ui_pyqt5.ui")[0]

class WindowClass(QMainWindow, form_class, KSPDeviceControl,Clustering,Prediction):
    def __init__(self):

        super().__init__()
        KSPDeviceControl.__init__(self)
        Clustering.__init__(self)

        '''
        ui connect
        '''

        self.setupUi(self)

        logo_pixmap = QPixmap(QtGui.QPixmap("/Images/NPL_logo.png"))
        self.NPLLOGO.setPixmap(logo_pixmap)

        self.MonitoringButton.clicked.connect(self.MonitoringFunction)
        self.ConnectButton.clicked.connect(self.ConnectFunction)
        self.DarkButton.clicked.connect(self.DarkFunction)
        self.SaveButton.clicked.connect(self.SaveFunction)

        '''
        Timer
        '''

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setGraphValue)
        self.timer_Active = False

        '''
        Dark Active
        '''

        self.dark_Active = False

        '''
        Monitoring Parameter Initialize
        '''
        self.TimeIndex = 0
        self.time_window = 10
        self.sampling_time = 0.5


    def ConnectFunction(self):
        try:
            self.Connect()
            '''
            Raw data Graph Initialize
            '''
            self.OriginalSeries = QLineSeries()

            self.OriginalChart = QChart()
            # self.OriginalChart.setAnimationOptions(QChart.AllAnimations)
            self.OriginalChart.createDefaultAxes()
            self.OriginalChart.legend().hide()
            self.OriginalChart.setTitle("Original Signal")
            self.OriginalChart.addSeries(self.OriginalSeries)

            self.OriginalXAxis = QValueAxis()
            self.OriginalXAxis.setRange(self.wl_table[0],self.wl_table[self.wl_real_num-1])
            self.OriginalChart.addAxis(self.OriginalXAxis, Qt.AlignBottom)
            self.OriginalSeries.attachAxis(self.OriginalXAxis)

            self.OriginalYAxis = QValueAxis()
            self.OriginalYAxis.setRange(0,5000)
            self.OriginalChart.addAxis(self.OriginalYAxis, Qt.AlignLeft)
            self.OriginalSeries.attachAxis(self.OriginalYAxis)

            self.OriginalSignal.setChart(self.OriginalChart)
            self.OriginalSignal.setRenderHints(QPainter.Antialiasing)

            '''
            Monitoring Graph Initialize
            '''

            self.MonitoringSeries = QLineSeries()

            self.MonitoringChart = QChart()
            self.MonitoringChart.setAnimationOptions(QChart.AllAnimations)
            self.MonitoringChart.createDefaultAxes()
            self.MonitoringChart.legend().hide()
            self.MonitoringChart.setTitle("Monitoring Signal")
            self.MonitoringChart.addSeries(self.MonitoringSeries)

            self.MonitoringXAxis = QValueAxis()
            self.MonitoringXAxis.setRange(self.time_window*self.sampling_time,10)
            self.MonitoringChart.addAxis(self.MonitoringXAxis, Qt.AlignBottom)
            self.MonitoringSeries.attachAxis(self.MonitoringXAxis)

            self.MonitoringYAxis = QValueAxis()
            self.MonitoringYAxis.setRange(0,1)
            self.MonitoringChart.addAxis(self.MonitoringYAxis, Qt.AlignLeft)
            self.MonitoringSeries.attachAxis(self.MonitoringYAxis)

            self.MonitoringSignal.setChart(self.MonitoringChart)
            self.MonitoringSignal.setRenderHints(QPainter.Antialiasing)

            self.OriginalData = np.empty(shape=(0,self.wl_real_num+1))
            self.MonitoringData = np.empty(shape=(0,2))
        except Exception as e:
            QMessageBox.critical(self,'Error Occured',str(e),QMessageBox.Ok)




    def MonitoringFunction(self):
        print("Button_1 clicked")
        try:
            if self.timer_Active:
                self.timer.stop()
                self.timer_Active = False

            else:
                self.TimeIndex = 0
                self.timer.start(int(self.sampling_time * 1000))
                self.timer_Active = True

        except Exception as e:
            QMessageBox.critical(self, 'Error Occured', str(e), QMessageBox.Ok)

    def DarkFunction(self):
        print("Button_2 clicked")
        try:
            if self.dark_Active:
                self.NAutoDark(0)
                self.dark_Active=False
            else:
                self.NAutoDark(1)
                self.dark_Active=True
        except Exception as e:
            QMessageBox.critical(self, 'Error Occured', str(e), QMessageBox.Ok)

    def SaveFunction(self):
        print("Save button clicked")
        try:
            np.savetxt(
                'test/test.csv',
                np.c_[
                    np.arange(self.sampling_time,self.TimeIndex*self.sampling_time,self.sampling_time),
                    self.OriginalData], delimiter=',',
            header=','.join(['Time(s)']+[wl for wl in self.wl_table_np])
            )
            np.savetxt(
                'test/monitoring.csv',
                np.c_[
                    np.arange(self.sampling_time*self.time_window,self.TimeIndex*self.sampling_time,self.sampling_time),
                    self.MonitoringData],delimiter=',',
                header="Time,Validity Score"
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error Occured', str(e), QMessageBox.Ok)

    def setGraphValue(self):
        data_np = np.frombuffer(self.NReadDataEx(),dtype=int)[:self.wl_real_num]
        self.OriginalData = np.r_[self.OriginalData,data_np.reshape(1,-1)]
        data_points = [QPointF(x,y) for x,y in zip(self.wl_table_np,data_np)]
        self.OriginalSeries.replace(data_points)
        self.OriginalSignal.update()

        self.WindowInput(data_np.reshape(1,-1))
        
        self.TimeIndex += 1
        if self.TimeIndex >= self.time_window:
            Validity = self.KMC()
            self.MonitoringSeries.append(QPointF(self.TimeIndex*self.sampling_time, Validity))
        if self.TimeIndex*0.5 > self.MonitoringXAxis.max():
            self.MonitoringXAxis.setMax(self.TimeIndex*self.sampling_time + 5)
        self.MonitoringSignal.update()

        if self.TimeIndex >=


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    myWindow.show()
    app.exec_()

