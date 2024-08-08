from PyQt6.QtCore import pyqtSignal, QObject
import numpy as np
import time
from rtlsdr import RtlSdr
from scipy.signal import firwin, bilinear, lfilter
import pyaudio
from demods.fm import fm_demod
 
class SDRWorker(QObject): # A QThread gets created in main_window which is assigned to this worker
    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run = pyqtSignal() # happens many times a second

    # Defaults
    fft_size = 4096
    buffer_size = int(100e3) # number of samples processed at a time for demod and such
    num_rows = 200
    #sample_rates = [3.2, 2.8, 2.56, 2.048, 1.2, 1.024] # MHz
    sample_rates = [1.024] # temporary, and in MHz
    time_plot_samples = 500
    gain = 50 # 0 to 73 dB. int

    # State
    spectrogram = -50*np.ones((fft_size, num_rows))
    PSD_avg = -50*np.ones(fft_size)
    demod_freq_khz = 0 # does not include center_freq
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
        # Init SDR
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rates[0]*1e6
        self.sdr.center_freq = 99.5e6
        self.sdr.gain = self.sdr.valid_gains_db[-1] # max gain
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
        self.sdr.center_freq = val*1e3
    
    def update_gain(self, val):
        print("Updated gain to:", self.sdr.valid_gains_db[val], 'dB')
        self.sdr.gain = self.sdr.valid_gains_db[val]

    def update_sample_rate(self, val):
        print("Updated sample rate to:", self.sample_rates[val], 'MHz')
        self.sdr.sample_rate = self.sample_rates[val] * 1e6
    
    def update_demod_freq(self, val):
        self.demod_freq_khz = val

    # Main loop
    def run(self):
        start_t = time.time()

        samples = self.sdr.read_samples(self.buffer_size)
        self.sample_count += len(samples)
        #print("Current rate:", self.sample_count/(time.time() - self.start_tt))

        #self.time_plot_update.emit(samples[0:time_plot_samples]/2**11) # make it go from -1 to 1 at highest gain
        self.time_plot_update.emit(samples[0:self.time_plot_samples])
        
        #PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2/(fft_size*self.sdr.sample_rate))
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[0:self.fft_size])))**2/self.fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)
    
        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)

        # Freq shift
        samples_shifted = samples * np.exp(-2j*np.pi*self.demod_freq_khz*1e3*np.arange(self.buffer_size)/self.sdr.sample_rate) # freq shift

        # Demod FM
        samples_demod, new_sample_rate = fm_demod(samples_shifted, self.sdr.sample_rate)

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
