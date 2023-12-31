# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
from pglive.sources.live_plot_widget import LivePlotWidget
from pglive.sources.live_axis_range import LiveAxisRange
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem


class ResizingView(QtWidgets.QGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.video_item = QGraphicsVideoItem()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.fitInView(self.video_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 500)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        # tab 1
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_logo1 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_logo1.setObjectName("verticalLayout_logo1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_logo1.addLayout(self.verticalLayout_3)
        self.horizontalLayout_logo1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_logo1.setObjectName("horizontalLayout_logo1")
        self.logo1 = QtWidgets.QLabel()
        self.horizontalLayout_logo1.addWidget(self.logo1)
        self.logo1b = QtWidgets.QLabel()
        self.horizontalLayout_logo1.addWidget(self.logo1b)
        self.verticalLayout_logo1.addLayout(self.horizontalLayout_logo1)
        self.pushButton_monitor = QtWidgets.QPushButton(parent=self.tab)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_monitor.setFont(font)
        self.pushButton_monitor.setObjectName("pushButton_monitor")
        self.verticalLayout_3.addWidget(self.pushButton_monitor)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.checkBox = QtWidgets.QCheckBox(parent=self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setMinimumSize(QtCore.QSize(70, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox.setFont(font)
        self.checkBox.setCheckable(True)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_7.addWidget(self.checkBox, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.label_11 = QtWidgets.QLabel(parent=self.tab)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_7.addWidget(self.label_11)
        self.label_model_prediction = QtWidgets.QLabel(parent=self.tab)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_model_prediction.setFont(font)
        self.label_model_prediction.setObjectName("label_model_prediction")
        self.horizontalLayout_7.addWidget(self.label_model_prediction)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.livePlotWidget = LivePlotWidget(parent=self.tab, y_range_controller=LiveAxisRange(fixed_range=[0, 25]))
        self.livePlotWidget.setAutoFillBackground(False)
        self.livePlotWidget.setObjectName("livePlotWidget")
        self.verticalLayout_3.addWidget(self.livePlotWidget)
        self.tabWidget.addTab(self.tab, "")

        # tab 2
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_logo2 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_logo2.setObjectName("verticalLayout_logo2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_logo2.addLayout(self.verticalLayout_2)
        self.horizontalLayout_logo2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_logo2.setObjectName("horizontalLayout_logo2")
        self.logo2 = QtWidgets.QLabel()
        self.horizontalLayout_logo2.addWidget(self.logo2)
        self.logo2b = QtWidgets.QLabel()
        sizePolicy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.logo2b.sizePolicy().hasHeightForWidth())
        self.logo2b.setSizePolicy(sizePolicy2)
        self.horizontalLayout_logo2.addWidget(self.logo2b)
        self.verticalLayout_logo2.addLayout(self.horizontalLayout_logo2)
        self.frame_2 = QtWidgets.QFrame(parent=self.tab_2)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_3.setMaximumSize(QtCore.QSize(500, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_duration = QtWidgets.QLineEdit(parent=self.frame_2)
        self.lineEdit_duration.setMinimumSize(QtCore.QSize(70, 0))
        self.lineEdit_duration.setMaximumSize(QtCore.QSize(500, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lineEdit_duration.setFont(font)
        self.lineEdit_duration.setClearButtonEnabled(False)
        self.lineEdit_duration.setObjectName("lineEdit_duration")
        self.horizontalLayout_3.addWidget(self.lineEdit_duration, QtCore.Qt.AlignmentFlag.AlignLeft)
        # self.horizontalLayout_3.addWidget(self.lineEdit_duration, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.pushButton_collector = QtWidgets.QPushButton(parent=self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.pushButton_collector.setFont(font)
        self.pushButton_collector.setObjectName("pushButton_collector")
        self.verticalLayout_2.addWidget(self.pushButton_collector)
        self.frame = QtWidgets.QFrame(parent=self.tab_2)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(parent=self.frame)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.label_timer = QtWidgets.QLabel(parent=self.frame)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_timer.setFont(font)
        self.label_timer.setObjectName("label_timer")
        self.horizontalLayout_2.addWidget(self.label_timer)
        self.verticalLayout_2.addWidget(self.frame)
        self.tabWidget.addTab(self.tab_2, "")

        # tab 3
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_logo3 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_logo3.setObjectName("verticalLayout_logo2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_logo3.addLayout(self.verticalLayout_6)
        self.horizontalLayout_logo3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_logo3.setObjectName("horizontalLayout_logo3")
        self.logo3 = QtWidgets.QLabel()
        self.horizontalLayout_logo3.addWidget(self.logo3)
        self.logo3b = QtWidgets.QLabel()
        self.horizontalLayout_logo3.addWidget(self.logo3b)
        self.verticalLayout_logo3.addLayout(self.horizontalLayout_logo3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton_load = QtWidgets.QPushButton(parent=self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_load.setFont(font)
        self.pushButton_load.setObjectName("pushButton_load")
        self.horizontalLayout_5.addWidget(self.pushButton_load)
        self.label_file = QtWidgets.QLabel(parent=self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_file.setFont(font)
        self.label_file.setObjectName("label_file")
        self.horizontalLayout_5.addWidget(self.label_file)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.pushButton_playback = QtWidgets.QPushButton(parent=self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_playback.setFont(font)
        self.pushButton_playback.setObjectName("pushButton_playback")
        self.verticalLayout.addWidget(self.pushButton_playback)
        self.verticalLayout_6.addLayout(self.verticalLayout)
        self.splitter_3 = QtWidgets.QSplitter(parent=self.tab_3)
        self.splitter_3.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter = QtWidgets.QSplitter(parent=self.splitter_3)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(parent=self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_6 = QtWidgets.QLabel(parent=self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(140, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setAutoFillBackground(False)
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_4.addWidget(self.label_6, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        # self.CSI_playback = PlotWidget(parent=self.layoutWidget)
        self.CSI_playback = LivePlotWidget(parent=self.layoutWidget, y_range_controller=LiveAxisRange(fixed_range=[0, 25]))
        self.CSI_playback.setAutoFillBackground(False)
        self.CSI_playback.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CSI_playback.sizePolicy().hasHeightForWidth())
        self.CSI_playback.setSizePolicy(sizePolicy)
        self.CSI_playback.setMinimumSize(QtCore.QSize(20, 0))
        self.CSI_playback.setObjectName("CSI_playback")
        self.verticalLayout_4.addWidget(self.CSI_playback)
        self.layoutWidget_2 = QtWidgets.QWidget(parent=self.splitter)
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_7 = QtWidgets.QLabel(parent=self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setMinimumSize(QtCore.QSize(140, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_5.addWidget(self.label_7, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        # self.video_widget = QVideoWidget(parent=self.layoutWidget_2)
        self.video_widget = ResizingView(parent=self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_widget.sizePolicy().hasHeightForWidth())
        self.video_widget.setSizePolicy(sizePolicy)
        self.video_widget.setMinimumSize(QtCore.QSize(20, 0))
        self.video_widget.setObjectName("video_widget")
        self.verticalLayout_5.addWidget(self.video_widget)
        self.splitter_2 = QtWidgets.QSplitter(parent=self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.layoutWidget_3 = QtWidgets.QWidget(parent=self.splitter_2)
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(parent=self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.label_pb_pred = QtWidgets.QLabel(parent=self.layoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_pb_pred.setFont(font)
        self.label_pb_pred.setObjectName("label_pb_pred")
        self.horizontalLayout.addWidget(self.label_pb_pred)
        self.layoutWidget_4 = QtWidgets.QWidget(parent=self.splitter_2)
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_8 = QtWidgets.QLabel(parent=self.layoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.label_pb_true = QtWidgets.QLabel(parent=self.layoutWidget_4)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_pb_true.setFont(font)
        self.label_pb_true.setObjectName("label_pb_true")
        self.horizontalLayout_4.addWidget(self.label_pb_true)
        self.verticalLayout_6.addWidget(self.splitter_3)
        self.tabWidget.addTab(self.tab_3, "")
        self.splitter.setSizes([300,300])
        self.splitter_2.setSizes([300,300])

        # main window
        self.horizontalLayout_6.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        # self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CSI HAR Demo"))
        self.pushButton_monitor.setText(_translate("MainWindow", "Start"))
        self.checkBox.setText(_translate("MainWindow", "Run Model"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">Prediction:</p></body></html>"))
        self.label_model_prediction.setText(_translate("MainWindow", "None"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "CSI Monitor"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">Experiment Duration:</p></body></html>"))
        self.lineEdit_duration.setPlaceholderText(_translate("MainWindow", "Enter experiment duration in seconds..."))
        self.pushButton_collector.setText(_translate("MainWindow", "Start"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">Timer:</p></body></html>"))
        self.label_timer.setText(_translate("MainWindow", "<html><head/><body><p>00:00</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "CSI Collector"))
        self.pushButton_load.setText(_translate("MainWindow", "Load File"))
        self.label_file.setText(_translate("MainWindow", "<html><head/><body><p>No file loaded</p></body></html>"))
        self.pushButton_playback.setText(_translate("MainWindow", "Start Playback"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">CSI Plotting</p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Experiment Video</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">Predited Gesture:</p></body></html>"))
        self.label_pb_pred.setText(_translate("MainWindow", "no gesture"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\">True Gesture:</p></body></html>"))
        self.label_pb_true.setText(_translate("MainWindow", "no gesture"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Experiment Viewer"))
