from PyQt6.QtCore import QSize, Qt, QThread, QTimer
from PyQt6.QtWidgets import QMainWindow, QGridLayout, QGraphicsRectItem, QWidget, QCheckBox, QSpacerItem, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QProgressBar  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
from sdr_thread import SDRWorker
import time
import sys

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #########
        # State #
        #########
        self.spectrogram_min = 0
        self.spectrogram_max = 0

        ############
        # GUI Init #
        ############
        self.setWindowTitle("HamWow")
        self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080
        layout = QGridLayout() # overall layout

        ###################
        # SDR Thread Init #
        ###################
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName('SDR_Thread') # so we can see it in htop, note you have to hit F2 -> Display options -> Show custom thread names
        worker = SDRWorker()
        self.worker = worker
        worker.moveToThread(self.sdr_thread)
        self.sdr_thread.started.connect(worker.run) # kicks off the worker when the thread starts

        ################
        # GUI Elements #
        ################

        # Time plot
        time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'})
        time_plot.setMouseEnabled(x=False, y=True)
        time_plot.setYRange(-1.1, 1.1)
        time_plot_curve_i = time_plot.plot([]) 
        time_plot_curve_q = time_plot.plot([]) 
        layout.addWidget(time_plot, 1, 0)

        # Time plot auto range buttons
        time_plot_auto_range_layout = QVBoxLayout()
        layout.addLayout(time_plot_auto_range_layout, 1, 1)
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : time_plot.autoRange()) # lambda just means its an unnamed function
        time_plot_auto_range_layout.addWidget(auto_range_button)
        auto_range_button2 = QPushButton('-1 to +1\n(ADC limits)')
        auto_range_button2.clicked.connect(lambda : time_plot.setYRange(-1.1, 1.1))
        time_plot_auto_range_layout.addWidget(auto_range_button2)

        # Freq plot
        freq_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'})
        freq_plot.setMouseEnabled(x=False, y=True)
        freq_plot_curve = freq_plot.plot([]) 
        freq_plot.setXRange(worker.sdr.center_freq/1e6 - worker.sdr.sample_rate/2e6, worker.sdr.center_freq/1e6 + worker.sdr.sample_rate/2e6)
        freq_plot.setYRange(-60, -20)
        layout.addWidget(freq_plot, 2, 0)
        
        # Freq auto range button
        auto_range_button = QPushButton('Auto Range')
        auto_range_button.clicked.connect(lambda : freq_plot.autoRange()) # lambda just means its an unnamed function
        layout.addWidget(auto_range_button, 2, 1)

        # Layout container for waterfall related stuff
        waterfall_layout = QHBoxLayout()
        layout.addLayout(waterfall_layout, 3, 0)

        # Waterfall plot
        waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'})
        imageitem = pg.ImageItem(axisOrder='col-major') # this arg is purely for performance
        waterfall.addItem(imageitem)
        waterfall.setMouseEnabled(x=False, y=False)
        waterfall_layout.addWidget(waterfall)

        # Colorbar for waterfall
        colorbar = pg.HistogramLUTWidget()
        colorbar.setImageItem(imageitem) # connects the bar to the waterfall imageitem
        colorbar.item.gradient.loadPreset('viridis') # set the color map, also sets the imageitem
        imageitem.setLevels((-60, -20)) # needs to come after colorbar is created for some reason
        waterfall_layout.addWidget(colorbar)

        # Waterfall auto range button
        auto_range_button = QPushButton('Auto Range\n(-2σ to +2σ)')
        def update_colormap():
            imageitem.setLevels((self.spectrogram_min, self.spectrogram_max))
            colorbar.setLevels(min=self.spectrogram_min, max=self.spectrogram_max)
        auto_range_button.clicked.connect(update_colormap)
        layout.addWidget(auto_range_button, 3, 1)

        layout.addWidget(QLabel('SDR Settings:'), 4, 0)

        # Freq slider with label, all units in kHz
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(int(13e3), int(1.75e6)) # in kHz
        freq_slider.setValue(int(worker.sdr.center_freq/1e3))
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        freq_slider.setTickInterval(int(1e6))
        freq_slider.valueChanged.connect(worker.update_freq)
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
            freq_plot.autoRange()
        freq_slider.valueChanged.connect(update_freq_label)
        update_freq_label(freq_slider.value()) # initialize the label
        layout.addWidget(freq_slider, 5, 0)
        layout.addWidget(freq_label, 5, 1)

        # Gain slider with label
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_slider.setTickInterval(1)
        gain_slider.valueChanged.connect(worker.update_gain)
        gain_slider.setRange(0, len(worker.sdr.valid_gains_db) - 1) # there's no easy way to make the interval not be 1... 
        gain_slider.setValue(len(worker.sdr.valid_gains_db) - 1) # highest gain index
        gain_label = QLabel()
        if not len(worker.sdr.valid_gains_db):
            print("Something went wrong")
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(worker.sdr.valid_gains_db[val]))
        update_gain_label(gain_slider.value()) # initialize the label
        gain_slider.valueChanged.connect(update_gain_label)
        layout.addWidget(gain_slider, 6, 0)
        layout.addWidget(gain_label, 6, 1)

        # SDR AGC checkbox
        agc_checkbox = QCheckBox("SDR AGC", self)
        agc_checkbox.stateChanged.connect(worker.update_agc)
        agc_checkbox.stateChanged.connect(lambda state: gain_slider.setEnabled(state == 0))
        layout.addWidget(agc_checkbox, 6, 2)

        # Sample rate dropdown using QComboBox
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in worker.sample_rates])
        sample_rate_combobox.setCurrentIndex(0) # should match the default at the top
        sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
        sample_rate_label = QLabel()
        def update_sample_rate_label(val):
            sample_rate_label.setText("Sample Rate: " + str(worker.sample_rates[val]) + " MHz")
        sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialize the label
        layout.addWidget(sample_rate_combobox, 7, 0)
        layout.addWidget(sample_rate_label, 7, 1)

        layout.addWidget(QLabel('Demod Settings:'), 8, 0)

        # Demod Freq slider with label, all units in kHz
        demod_freq_slider = QSlider(Qt.Orientation.Horizontal)
        demod_freq_slider.setRange(int(-2e3), int(2e3)) # in kHz, based on max sample rate
        demod_freq_slider.setValue(0)
        demod_freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        demod_freq_slider.setTickInterval(int(0.1e3))
        demod_freq_slider.valueChanged.connect(worker.update_demod_freq)
        demod_freq_label = QLabel()
        rect_item = QGraphicsRectItem()
        rect_item.setBrush(pg.mkBrush((0, 0, 255, 100)))
        rect_item.setPen(pg.mkPen((0, 0, 0), width=0))
        freq_plot.addItem(rect_item)
        demod_line = freq_plot.addLine(x=100, y=None, pen={'color':'green', 'width':1.5}) # Demod line on freq_plot
        def update_demod_freq_label(val):
            demod_freq_label.setText("Demod Frequency [kHz]: " + str(val))
            demod_line.setValue(freq_slider.value()/1e3 + val/1e3)
            rect_item.setRect(freq_slider.value()/1e3 + val/1e3, -100, 0.05, 200) #x,y,width,height
        demod_freq_slider.valueChanged.connect(update_demod_freq_label)
        update_demod_freq_label(demod_freq_slider.value()) # initialize the label
        layout.addWidget(demod_freq_slider, 9, 0)
        layout.addWidget(demod_freq_label, 9, 1)

        # Realtime ratio bar
        progress_bar_layout = QHBoxLayout()
        layout.addLayout(progress_bar_layout, 10, 0)
        progress_bar = QProgressBar()
        progress_bar_layout.addWidget(QLabel("Realtime Ratio:"))
        progress_bar_layout.addWidget(progress_bar)
        progress_bar_layout.addItem(QSpacerItem(1000, 10))

        # Buffer indicator bar
        buffer_bar_layout = QHBoxLayout()
        layout.addLayout(buffer_bar_layout, 11, 0)
        buffer_bar = QProgressBar()
        buffer_bar_layout.addWidget(QLabel("Buffer Fullness:"))
        buffer_bar_layout.addWidget(buffer_bar)
        buffer_bar_layout.addItem(QSpacerItem(1000, 10))

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        #####################
        # Signals and Slots #
        #####################

        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)
        
        def freq_plot_callback(PSD_avg):
            # TODO figure out if there's a way to just change the visual ticks instead of the actual x vals
            f = np.linspace(freq_slider.value()*1e3 - worker.sdr.sample_rate/2.0, freq_slider.value()*1e3 + worker.sdr.sample_rate/2.0, worker.fft_size) / 1e6
            freq_plot_curve.setData(f, PSD_avg)
            freq_plot.setXRange(freq_slider.value()*1e3/1e6 - worker.sdr.sample_rate/2e6, freq_slider.value()*1e3/1e6 + worker.sdr.sample_rate/2e6)
        
        def waterfall_plot_callback(spectrogram):
            imageitem.setImage(spectrogram, autoLevels=False)
            sigma = np.std(spectrogram)
            mean = np.mean(spectrogram) 
            self.spectrogram_min = mean - 2*sigma # save to window state
            self.spectrogram_max = mean + 2*sigma

        # connect the signal to the callback
        worker.time_plot_update.connect(time_plot_callback) 
        worker.freq_plot_update.connect(freq_plot_callback)
        worker.waterfall_plot_update.connect(waterfall_plot_callback)
        worker.progress_bar_update.connect(lambda x: progress_bar.setValue(int(x * 100)))
        worker.buffer_bar_update.connect(lambda x: buffer_bar.setValue(x))
        worker.end_of_run.connect(lambda : QTimer.singleShot(0, worker.run)) # Run worker again immediately

        self.sdr_thread.start() # not blocking

    def closeEvent(self, event): # gets called when you close the window
        self.worker.end_of_run.disconnect() # stop running the run() function
        #self.worker.end_of_run.connect(self.sdr_thread.quit) # quit the thread when the run() function is done
        time.sleep(0.3) # give time for the next run() to finish
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait()

