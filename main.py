from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QSlider, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox  # tested with PyQt6==6.7.0
import pyqtgraph as pg # tested with pyqtgraph==0.13.7
import numpy as np
import time
import signal # lets control-C actually close the app
from rtlsdr import RtlSdr
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import pyaudio

# Defaults
fft_size = 4096
buffer_size = int(100e3) # number of samples processed at a time for demod and such
num_rows = 200
center_freq = 99.5e6
#sample_rates = [3.2, 2.8, 2.56, 2.048, 1.2, 1.024] # MHz
sample_rates = [1.024] # MHz
sample_rate = sample_rates[0] * 1e6
time_plot_samples = 500
gain = 50 # 0 to 73 dB. int

# Init SDR
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = sdr.valid_gains_db[-1] # max gain

class SDRWorker(QObject):
    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # happens many times a second

    # State
    spectrogram = -50*np.ones((fft_size, num_rows))
    PSD_avg = -50*np.ones(fft_size)
    demod_freq_khz = 0 # does not include center_freq
    bz, az = bilinear(1, [75e-6, 1], fs=sample_rate) # De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
    h_lowpass = firwin(51, cutoff=50e3, fs=sample_rate)
    sample_count = 0
    start_tt = time.time()
    audio_buffer_read_pointer = 0
    audio_buffer_write_pointer = 0
    audio_buffer = np.zeros(int(100e6), dtype=np.float32)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        #print("GOT HERE, frame_count:", frame_count)
        # If len(data) is less than requested frame_count, PyAudio automatically assumes the stream is finished, and the stream stops.
        if self.audio_buffer_read_pointer > (self.audio_buffer_write_pointer + frame_count):
            print("Waiting for more audio data")
            time.sleep(1)
            #return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue) # output zeros
        samples_to_play = self.audio_buffer[self.audio_buffer_read_pointer:self.audio_buffer_read_pointer+(frame_count)]
        self.audio_buffer_read_pointer += (frame_count)
        #print("Read pointer:", self.audio_buffer_read_pointer)
        output_bytes = samples_to_play.tobytes() # explicitly convert to bytes sequence, sample values must be in range [-1.0, 1.0]
        return (output_bytes, pyaudio.paContinue)

    def __init__(self):
        super(SDRWorker, self).__init__()
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=42666,
                             output=True,
                             frames_per_buffer=100, # determines how many samples the callback function asks for each time
                             stream_callback=self.audio_callback)
    # PyQt Slots
    def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
        print("Updated freq to:", val, 'kHz')
        #sdr.rx_lo = int(val*1e3)
        #usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(val*1e3), 0)
        #flush_buffer()
        sdr.center_freq = val*1e3
    
    def update_gain(self, val):
        print("Updated gain to:", sdr.valid_gains_db[val], 'dB')
        #sdr.rx_hardwaregain_chan0 = val
        #usrp.set_rx_gain(val, 0)
        #flush_buffer()
        sdr.gain = sdr.valid_gains_db[val]

    def update_sample_rate(self, val):
        print("Updated sample rate to:", sample_rates[val], 'MHz')
        #sdr.sample_rate = int(sample_rates[val] * 1e6)
        #sdr.rx_rf_bandwidth = int(sample_rates[val] * 1e6 * 0.8)
        #usrp.set_rx_rate(sample_rates[val] * 1e6, 0)
        #flush_buffer()
        sdr.sample_rate = sample_rates[val] * 1e6
    
    def update_demod_freq(self, val):
        self.demod_freq_khz = val

    # Main loop
    def run(self):
        start_t = time.time()
                
        #samples = sdr.rx() # Receive samples
        #streamer.recv(recv_buffer, metadata)
        #samples = recv_buffer[0] # will be np.complex64
        samples = sdr.read_samples(buffer_size)
        self.sample_count += len(samples)
        #print("Current rate:", self.sample_count/(time.time() - self.start_tt))

        #self.time_plot_update.emit(samples[0:time_plot_samples]/2**11) # make it go from -1 to 1 at highest gain
        self.time_plot_update.emit(samples[0:time_plot_samples])
        
        #PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/(fft_size*sample_rate))
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[0:fft_size])))**2/fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)
    
        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)

        # Demod FM
        samples_demod = samples * np.exp(-2j*np.pi*self.demod_freq_khz*1e3*np.arange(buffer_size)/sample_rate) # freq shift
        samples_demod = np.convolve(samples_demod, self.h_lowpass, 'same') # Low pass filter
        samples_demod = samples_demod[::4] # decimate, simply to zoom into the signal
        samples_demod = np.diff(np.unwrap(np.angle(samples_demod))) # FM demod
        samples_demod = lfilter(self.bz, self.az, samples_demod) # apply de-emphasis filter
        samples_demod = samples_demod[::6] # decimate by 6 to get mono audio
        sample_rate_audio = sdr.sample_rate/6/4
        #print("Sample rate audio:", sample_rate_audio)

        #h_audio = firwin(21, cutoff=2e3, fs=sample_rate_audio)
        #samples_demod = np.convolve(samples_demod, h_audio, 'same') # Low pass filter to get rid of popping
        
        # Play audio
        #samples_demod /= np.max(np.abs(samples_demod)) # normalize volume so its between -1 and +1
        samples_demod *= 0.5
        if np.max(np.abs(samples_demod)) > 1:
            print("Audio saturated")
            samples_demod *= 0.5
        samples_demod = samples_demod.astype(np.float32)
        #print(len(samples_demod))
        #output_bytes = samples_demod.tobytes() # explicitly convert to bytes sequence, sample values must be in range [-1.0, 1.0]
        #stream.write(output_bytes)
        self.audio_buffer[self.audio_buffer_write_pointer:self.audio_buffer_write_pointer+len(samples_demod)] = samples_demod
        self.audio_buffer_write_pointer += len(samples_demod)
        #print("Write pointer:", self.audio_buffer_write_pointer)

        # Hacky way to deal with buffer filling up, which will happen after several minutes
        if self.audio_buffer_write_pointer > len(self.audio_buffer) - len(samples_demod):
            print("Audio buffer full, resetting read and write pointer")
            self.audio_buffer_write_pointer = 0
            self.audio_buffer_read_pointer = 0

        #print("Frames per second:", 1/(time.time() - start_t))
        self.end_of_run.emit() # emit the signal to keep the loop going


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HamWow")
        self.setFixedSize(QSize(1500, 1000)) # window size, starting size should fit on 1920 x 1080

        self.spectrogram_min = 0
        self.spectrogram_max = 0

        layout = QGridLayout() # overall layout

        # Initialize worker and thread
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName('SDR_Thread') # so we can see it in htop, note you have to hit F2 -> Display options -> Show custom thread names
        worker = SDRWorker()
        worker.moveToThread(self.sdr_thread)

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
        freq_plot.setXRange(center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6)
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
        freq_slider.setValue(int(center_freq/1e3))
        freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        freq_slider.setTickInterval(int(1e6))
        freq_slider.sliderMoved.connect(worker.update_freq) # there's also a valueChanged option
        freq_label = QLabel()
        def update_freq_label(val):
            freq_label.setText("Frequency [MHz]: " + str(val/1e3))
            freq_plot.autoRange()
        freq_slider.sliderMoved.connect(update_freq_label)
        update_freq_label(freq_slider.value()) # initialize the label
        layout.addWidget(freq_slider, 5, 0)
        layout.addWidget(freq_label, 5, 1)

        # Gain slider with label
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, len(sdr.valid_gains_db) - 1) # there's no easy way to make the interval not be 1... 
        gain_slider.setValue(len(sdr.valid_gains_db) - 1) # highest gain index
        gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_slider.setTickInterval(1)
        gain_slider.sliderMoved.connect(worker.update_gain)
        gain_label = QLabel()
        def update_gain_label(val):
            gain_label.setText("Gain: " + str(sdr.valid_gains_db[val]))
        gain_slider.sliderMoved.connect(update_gain_label)
        update_gain_label(gain_slider.value()) # initialize the label
        layout.addWidget(gain_slider, 6, 0)
        layout.addWidget(gain_label, 6, 1)

        # Sample rate dropdown using QComboBox
        sample_rate_combobox = QComboBox()
        sample_rate_combobox.addItems([str(x) + ' MHz' for x in sample_rates])
        sample_rate_combobox.setCurrentIndex(0) # should match the default at the top
        sample_rate_combobox.currentIndexChanged.connect(worker.update_sample_rate)
        sample_rate_label = QLabel()
        def update_sample_rate_label(val):
            sample_rate_label.setText("Sample Rate: " + str(sample_rates[val]) + " MHz")
        sample_rate_combobox.currentIndexChanged.connect(update_sample_rate_label)
        update_sample_rate_label(sample_rate_combobox.currentIndex()) # initialize the label
        layout.addWidget(sample_rate_combobox, 7, 0)
        layout.addWidget(sample_rate_label, 7, 1)

        layout.addWidget(QLabel('Demod Settings:'), 8, 0)

        # Demod line on freq_plot
        demod_line = freq_plot.addLine(x=100, y=None, pen={'color':'green', 'width':1.5})

        # Demod Freq slider with label, all units in kHz
        demod_freq_slider = QSlider(Qt.Orientation.Horizontal)
        demod_freq_slider.setRange(int(-2e3), int(2e3)) # in kHz, based on max sample rate
        demod_freq_slider.setValue(0)
        demod_freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        demod_freq_slider.setTickInterval(int(0.1e3))
        demod_freq_slider.sliderMoved.connect(worker.update_demod_freq)
        demod_freq_label = QLabel()
        def update_demod_freq_label(val):
            demod_freq_label.setText("Demod Frequency [kHz]: " + str(val))
            demod_line.setValue(freq_slider.value()/1e3 + val/1e3)
        demod_freq_slider.sliderMoved.connect(update_demod_freq_label)
        update_demod_freq_label(demod_freq_slider.value()) # initialize the label
        layout.addWidget(demod_freq_slider, 9, 0)
        layout.addWidget(demod_freq_label, 9, 1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Signals and slots stuff
        def time_plot_callback(samples):
            time_plot_curve_i.setData(samples.real)
            time_plot_curve_q.setData(samples.imag)
        
        def freq_plot_callback(PSD_avg):
            # TODO figure out if there's a way to just change the visual ticks instead of the actual x vals
            f = np.linspace(freq_slider.value()*1e3 - sample_rate/2.0, freq_slider.value()*1e3 + sample_rate/2.0, fft_size) / 1e6
            freq_plot_curve.setData(f, PSD_avg)
            freq_plot.setXRange(freq_slider.value()*1e3/1e6 - sample_rate/2e6, freq_slider.value()*1e3/1e6 + sample_rate/2e6)
        
        def waterfall_plot_callback(spectrogram):
            imageitem.setImage(spectrogram, autoLevels=False)
            sigma = np.std(spectrogram)
            mean = np.mean(spectrogram) 
            self.spectrogram_min = mean - 2*sigma # save to window state
            self.spectrogram_max = mean + 2*sigma

        def end_of_run_callback():
            QTimer.singleShot(0, worker.run) # Run worker again immediately
        
        worker.time_plot_update.connect(time_plot_callback) # connect the signal to the callback
        worker.freq_plot_update.connect(freq_plot_callback)
        worker.waterfall_plot_update.connect(waterfall_plot_callback)
        worker.end_of_run.connect(end_of_run_callback)

        self.sdr_thread.started.connect(worker.run) # kicks off the worker when the thread starts
        self.sdr_thread.start()


app = QApplication([])
window = MainWindow()
window.show() # Windows are hidden by default
signal.signal(signal.SIGINT, signal.SIG_DFL) # this lets control-C actually close the app
app.exec() # Start the event loop

#stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
#streamer.issue_stream_cmd(stream_cmd)
