import sys, os
from threading import Thread, Event
from statistics import mode
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtMultimedia import QMediaPlayer, QMediaFormat
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
import qdarktheme
import pyqtgraph as pg
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget
from MainWindow import Ui_MainWindow
from read_stdin import readline, print_until_first_csi_line
from real_time_csi import *


class CSIMonitorOut(QtCore.QObject):
    # class for signals connecting CSI monitor to GUI
    prediction = QtCore.pyqtSignal(str)


class ReplayOut(QtCore.QObject):
    # class for signals connecting replay outputs to GUI
    outputs = QtCore.pyqtSignal(str, str)


class TimerOut(QtCore.QObject):
    # class for signals connecting timer to GUI
    timer = QtCore.pyqtSignal(int)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # class for GUI window
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/figs/Utoronto_coa.png"))


        ### General Use

        # thread to keep stdin from building up
        self.event_stop_clear = Event()
        self.thread_clear_stdin = Thread(target=clear_stdin, args=(self.event_stop_clear,))
        self.thread_clear_stdin.start()


        ### Tab 1: CSI Monitor
        
        # logo on tab 1
        self.logo1b.setPixmap(QtGui.QPixmap("/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/figs/ECE_Signature.svg"))
        self.logo1b.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.logo1.setText("WIRLab")
        self.logo1.setStyleSheet("QLabel {color: #25355A}")
        self.logo1.setFont(QtGui.QFont("Helvetica", 20, 900))
        self.logo1.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        # push button to start/stop csi monitor
        self.pushButton_monitor.setCheckable(True)
        self.pushButton_monitor.clicked.connect(self.push_csi_monitor_button)
        self.pushButton_monitor.setStyleSheet(
            'QPushButton {border-style: outset; border-width: 2px; border-color: #25355A;}')

        # live CSI plot
        self.livePlotWidget.setBackground('w')
        self.livePlotWidget.showGrid(x=True, y=True)
        self.pen = pg.mkPen(color=(37, 53, 90), width=1)
        self.plot_curve = LiveLinePlot(pen=self.pen)
        self.livePlotWidget.addItem(self.plot_curve)
        self.data_connector = DataConnector(self.plot_curve, max_points=800, update_rate=300)

        # signal to connect output of model to GUI
        self.model_output = CSIMonitorOut()
        self.model_output.prediction.connect(self.display_model_prediction)

        # event to trigger the use of model on captured CSI data
        self.use_model = Event()
        self.checkBox.stateChanged.connect(self.set_use_model)


        ### Tab 2: CSI Collector

        # logo on tab 2
        self.logo2b.setPixmap(QtGui.QPixmap("/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/figs/ECE_Signature.svg"))
        self.logo2b.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.logo2.setText("WIRLab")
        self.logo2.setStyleSheet("QLabel {color: #25355A}")
        self.logo2.setFont(QtGui.QFont("Helvetica", 20, 900))
        self.logo2.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        # push button to start/stop collection
        self.pushButton_collector.setCheckable(True)
        self.pushButton_collector.clicked.connect(self.push_csi_collect_button)
        self.pushButton_collector.setStyleSheet(
            'QPushButton {border-style: outset; border-width: 2px; border-color: #25355A;}')

        # timer functionality
        self.timer_output = TimerOut()
        self.timer_output.timer.connect(self.display_timer)
        self.experiment_time_complete = TimerOut()
        self.experiment_time_complete.timer.connect(self.collection_time_complete)


        ### Tab 3: Experiment Replay

        # logo on tab 3
        self.logo3b.setPixmap(QtGui.QPixmap("/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/figs/ECE_Signature.svg"))
        self.logo3b.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.logo3.setText("WIRLab")
        self.logo3.setStyleSheet("QLabel {color: #25355A}")
        self.logo3.setFont(QtGui.QFont("Helvetica", 20, 900))
        self.logo3.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        # load button and file label
        self.pushButton_load.clicked.connect(self.load_file)
        self.pushButton_load.setStyleSheet(
            'QPushButton {border-style: outset; border-width: 2px; border-color: #25355A;}')

        # start playback button
        self.pushButton_playback.clicked.connect(self.start_playback)
        self.pushButton_playback.setStyleSheet(
            'QPushButton {border-style: outset; border-width: 2px; border-color: #25355A;}')

        # CSI plotting playback
        self.replay_csi = "/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/final_demo_data/bedroom_demo_1.mat"
        self.CSI_playback.setBackground('w')
        self.CSI_playback.showGrid(x=True, y=True)
        self.pen2 = pg.mkPen(color=(37, 53, 90), width=1)
        self.plot_replay = LiveLinePlot(pen=self.pen2)
        self.CSI_playback.addItem(self.plot_replay)
        self.data_replay = DataConnector(self.plot_replay, max_points=800, update_rate=300)
        self.event_stop_replay = None
        self.thread_replay = None

        # experiment video player
        self.scene = QtWidgets.QGraphicsScene()
        self.media_player = QMediaPlayer()

        self.scene.setBackgroundBrush(QtGui.QColor.fromRgb(50,50,50))
        self.scene.addItem(self.video_widget.video_item)
        self.video_widget.setScene(self.scene)
        self.replay_video = "/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/final_demo_data/bedroom_demo_1.mp4"
        self.media_player.setSource(QtCore.QUrl.fromLocalFile(self.replay_video))
        self.media_player.setVideoOutput(self.video_widget.video_item)

        # labels for predicted and true gestures
        self.replay_preds = "/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/final_demo_data/bedroom_demo_1.csv"
        self.replay_outputs = ReplayOut()
        self.replay_outputs.outputs.connect(self.display_replay)
        # self.label_pb_pred
        # self.label_pb_true


    def push_csi_monitor_button(self, checked):
        if checked:
            self.pushButton_monitor.setText("Stop")

            # stop clearing the stdin
            self.event_stop_clear.set()
            self.thread_clear_stdin.join()

            # start new thread to run real-time monitoring
            self.event_stop_csi = Event()
            self.live_plot_thread = Thread(target=real_time, args=(self.data_connector,
                                                                   self.model_output,
                                                                   self.event_stop_csi,
                                                                   self.use_model,
                                                                   True, False, 9999,))
            self.live_plot_thread.start()
        else:
            self.pushButton_monitor.setText("Start")

            # stop real-time monitoring thread
            self.event_stop_csi.set()
            self.live_plot_thread.join()

            # start new stdin clearing thread
            self.event_stop_clear.clear()
            self.thread_clear_stdin = Thread(target=clear_stdin, args=(self.event_stop_clear,))
            self.thread_clear_stdin.start()

    @QtCore.pyqtSlot(str)
    def display_model_prediction(self, prediction):
        self.label_model_prediction.setText(prediction)

    @QtCore.pyqtSlot(int)
    def set_use_model(self, state):
        if state == 2:
            self.use_model.set()
        elif state == 0:
            self.use_model.clear()

    def push_csi_collect_button(self, checked):
        if checked:
            self.collection_duration = 0
            if self.lineEdit_duration.text() == "":
                self.pushButton_collector.setText("Stop")
                self.label_timer.setText("00:00")
                self.collection_duration = 9999
            else:
                try:
                    self.collection_duration = int(self.lineEdit_duration.text())
                    if self.collection_duration > 0:
                        self.pushButton_collector.setText("Stop")
                        self.label_timer.setText("00:00")
                    else:
                        self.label_timer.setText("Invalid Duration")
                        self.pushButton_collector.setChecked(False)
                except ValueError:
                    self.label_timer.setText("Invalid Duration")
                    self.pushButton_collector.setChecked(False)

            if self.collection_duration > 0:
                # stop clearing the stdin
                self.event_stop_clear.set()
                self.thread_clear_stdin.join()

                # start new thread to run real-time monitoring
                self.event_stop_csi_collection = Event()
                self.use_model.clear()
                self.live_collect_thread = Thread(target=real_time, args=(self.data_connector,
                                                                    self.timer_output,
                                                                    self.event_stop_csi_collection,
                                                                    self.use_model,
                                                                    False, True,
                                                                    self.collection_duration,
                                                                    self.experiment_time_complete,))
                self.live_collect_thread.start()
        else:
            self.pushButton_collector.setText("Start")
            # self.label_timer.setText("DONE")

            # stop real-time monitoring thread
            self.event_stop_csi_collection.set()
            self.live_collect_thread.join()

            # start new stdin clearing thread
            self.event_stop_clear.clear()
            self.thread_clear_stdin = Thread(target=clear_stdin, args=(self.event_stop_clear,))
            self.thread_clear_stdin.start()

    @QtCore.pyqtSlot(int)
    def display_timer(self, timer):
        #convert int to nice time formatting (xx:yy for mins and secs)
        minutes = int(timer / 60)
        seconds = timer % 60
        self.label_timer.setText(f"{minutes:02d}:{seconds:02d}")

    @QtCore.pyqtSlot(int)
    def collection_time_complete(self, timer):
        if timer == 1:
            self.pushButton_collector.setText("Start")
            self.label_timer.setText("DONE")
            self.pushButton_collector.setChecked(False)

            # stop real-time monitoring thread
            self.event_stop_csi_collection.set()
            self.live_collect_thread.join()

            # start new stdin clearing thread
            self.event_stop_clear.clear()
            self.thread_clear_stdin = Thread(target=clear_stdin, args=(self.event_stop_clear,))
            self.thread_clear_stdin.start()

    def load_file(self):
        self.media_player.pause()
        if self.thread_replay != None:
            if self.thread_replay.is_alive():
                self.event_stop_replay.set()
                self.thread_replay.join()
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "/home/raso/esp/esp32-csi-tool/python_utils/demo_gui/final_demo_data/",
            "Video Files (*.mp4 *.MOV)", # "All Files (*)",
        )
        if fname[0] != "":
            self.replay_video = fname[0]
            self.media_player.setSource(QtCore.QUrl.fromLocalFile(self.replay_video))
            
            directory, filename = os.path.split(fname[0])
            self.label_file.setText(filename)

            # find CSI data file with same experiment name
            experiment_name = filename.split(".")[0]
            if os.path.isfile(os.path.join(directory, experiment_name + ".mat")):
                self.replay_csi = os.path.join(directory, experiment_name + ".mat")
                # print(os.path.split(self.replay_csi)[1])
            else: print("ERROR: No matching CSI file found")

            # find csv data file with same experiment name
            if os.path.isfile(os.path.join(directory, experiment_name + ".csv")):
                self.replay_preds = os.path.join(directory, experiment_name + ".csv")
                # print(os.path.split(self.replay_csi)[1])
            else: print("ERROR: No matching csv file found")

    def start_playback(self):
        # if playing, stop and restart thread as well
        if self.thread_replay != None:
            if self.thread_replay.is_alive():
                self.event_stop_replay.set()
                self.thread_replay.join()
        self.data_replay.cb_set_data([0], [0])

        # start video playback
        self.media_player.setPosition(0)#5300)
        self.media_player.play()
        self.video_widget.fitInView(self.video_widget.video_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        # start new thread to run real-time monitoring
        self.event_stop_replay = Event()
        self.thread_replay = Thread(target=csi_playback, args=(self.replay_csi,
                                                               self.data_replay,
                                                               self.replay_preds,
                                                               self.event_stop_replay,
                                                               self.replay_outputs,))
        self.thread_replay.start()

    @QtCore.pyqtSlot(str, str)
    def display_replay(self, prediction, true_class):
        if prediction == true_class:
            self.label_pb_pred.setStyleSheet('color: green')
        else: self.label_pb_pred.setStyleSheet('color: red')
        self.label_pb_pred.setText(prediction)
        self.label_pb_true.setText(true_class)


def main():
    # load all parts necessary for CSI and HAR
    # print_until_first_csi_line()
    print("Start")

    # load app
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme(
        theme = "light",
        corner_shape= "rounded",
        custom_colors={
            "[light]": {
                "background": "#F2F4F7",
                "primary": "#25355A",
            }
        }
    )

    main = MainWindow()
    main.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
