from PyQt6.QtCore import pyqtSignal, QObject
import numpy as np
import time
from rtlsdr import RtlSdr
import pyaudio
from demods.fm import fm_demod
from demods.dsb_am import dsb_am
from threading import Thread
import asyncio
from dsp.agc import AGC

class SDRWorker(QObject): # A QThread gets created in main_window which is assigned to this worker
    # PyQt Signals
    time_plot_update = pyqtSignal(np.ndarray)
    freq_plot_update = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    progress_bar_update = pyqtSignal(float)
    buffer_bar_update = pyqtSignal(int)
    end_of_run = pyqtSignal() # happens many times a second

    # Defaults
    fft_size = 4096
    buffer_size = int(100e3) # requested (not always met perfectly) number of samples processed at a time for demod and such
    samples_batches = []
    num_rows = 200
    sample_rates = [1.024, 3.2, 2.8, 2.56, 2.048, 1.2] # MHz
    time_plot_samples = 500
    gain = 50 # 0 to 73 dB. int
    direct_sampling = 0 # 0 for superhet, 2 or direct

    # State
    spectrogram = -50*np.ones((fft_size, num_rows))
    PSD_avg = -50*np.ones(fft_size)
    demod_freq_khz = 0 # does not include center_freq
    demod_type = 'WFM' # WFM, DSB AM, USB AM, LSB AM, CW
    sample_read_timer = time.time()
    realtime_ratio = 0
    audio_buffer_read_pointer = 0
    audio_buffer_write_pointer = 0
    audio_buffer = np.zeros(int(100e6), dtype=np.float32)
    kill_signal = False
    gain = -1 # holds gain for when AGC is turned off
    
    # Note- this gets called automatically by PyAudio when the stream is started, and it doesnt block the run() calls
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

    async def rtl_callback(self):
        async for samples in self.sdr.stream():
            if self.kill_signal:
                print("Killing RTL callback")
                break
            if len(self.samples_batches) < 100: # keep this at 100 so it aligns with the buffer bar max value of 100
                self.samples_batches.append(samples)
            else:
                print("Samples buffer full, either window was closed, or sample rate is too high for amount of DSP")
        self.sdr.stop()
        
    def rtl_thread_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.rtl_callback()) # blocking

    def __init__(self):
        super(SDRWorker, self).__init__()
        # Init SDR
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rates[0]*1e6
        self.sdr.center_freq = 99.5e6
        self.sdr.gain = self.sdr.valid_gains_db[-1] # max gain
        self.rtl_thread = Thread(target = self.rtl_thread_worker, args = ())
        self.rtl_thread.start()

        # Init audio
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=42666,
                             output=True,
                             frames_per_buffer=100, # determines how many samples the callback function asks for each time
                             stream_callback=self.audio_callback)
        
        # Init DSP
        self.fm_demod = fm_demod(self.sdr.sample_rate)
        self.dsb_am_demod = dsb_am(self.sdr.sample_rate)
        
        # AGC
        self.agc = AGC(self.sdr.sample_rate/24, 10, 20, 0.5)

    # PyQt Slots
    def update_freq(self, val): # TODO: WE COULD JUST MODIFY THE SDR IN THE GUI THREAD
        print("Updated freq to:", val, 'kHz')
        self.sdr.center_freq = val*1e3
    
    def update_gain(self, val):
        print("Updated gain to:", self.sdr.valid_gains_db[val], 'dB')
        self.sdr.gain = self.sdr.valid_gains_db[val]
        self.gain = val

    def update_sample_rate(self, val):
        print("Updated sample rate to:", self.sample_rates[val], 'MHz')
        self.sdr.sample_rate = self.sample_rates[val] * 1e6
    
    def update_demod_freq(self, val):
        self.demod_freq_khz = val
        
    def update_demod_type(self, val):
        self.demod_type = val
        print("Updated demod type to:", val)

    def update_agc(self, val):
        if val:
            print("Turning AGC on")
            self.sdr.gain = 'auto'
        else:
            print("Turning AGC off")
            self.sdr.gain = self.sdr.valid_gains_db[self.gain]
    
    def update_dir_sampling(self, val):
        self.direct_sampling = val*2
        self.sdr.set_direct_sampling(val*2)

    # Main loop
    def run(self):
        start_t = time.time()

        # Wait until a batch of samples is available to process, only process 1 batch at a time, per call to run()
        while len(self.samples_batches) == 0:
            time.sleep(0.01)
        self.buffer_bar_update.emit(len(self.samples_batches)) # update buffer bar
        samples = self.samples_batches.pop(0) # grab oldest batch of samples
        
        self.realtime_ratio = len(samples) / ((time.time() - self.sample_read_timer) * self.sdr.sample_rate)
        self.sample_read_timer = time.time()
        self.progress_bar_update.emit(self.realtime_ratio)
        
        self.time_plot_update.emit(samples[0:self.time_plot_samples])
        
        PSD = 10.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[0:self.fft_size])))**2/self.fft_size)
        self.PSD_avg = self.PSD_avg * 0.99 + PSD * 0.01
        self.freq_plot_update.emit(self.PSD_avg)
    
        self.spectrogram[:] = np.roll(self.spectrogram, 1, axis=1) # shifts waterfall 1 row
        self.spectrogram[:,0] = PSD # fill last row with new fft results
        self.waterfall_plot_update.emit(self.spectrogram)

        # Freq shift
        samples_shifted = samples * np.exp(-2j*np.pi*self.demod_freq_khz*1e3*np.arange(len(samples))/self.sdr.sample_rate) # freq shift

        # Demod using selected type
        if self.demod_type == 'WFM': # WFM
            samples_demod, new_sample_rate = self.fm_demod.process(samples_shifted)
        elif self.demod_type == 'DSB AM': # DSB AM
            samples_demod, new_sample_rate = self.dsb_am_demod.process(samples_shifted)
        
        # Apply audio AGC
        samples_demod = self.agc.apply_agc(samples_demod)

        # Play audio
        #samples_demod /= np.max(np.abs(samples_demod)) # normalize volume so its between -1 and +1
        samples_demod *= 0.5
        if np.max(np.abs(samples_demod)) > 1:
            print("Audio saturated")
            samples_demod *= 0.5
        samples_demod = samples_demod.astype(np.float32)
        self.audio_buffer[self.audio_buffer_write_pointer:self.audio_buffer_write_pointer+len(samples_demod)] = samples_demod
        self.audio_buffer_write_pointer += len(samples_demod)

        # Hacky way to deal with buffer filling up, which will happen after several minutes
        if self.audio_buffer_write_pointer > len(self.audio_buffer) - len(samples_demod):
            print("Audio buffer full, resetting read and write pointer")
            self.audio_buffer_write_pointer = 0
            self.audio_buffer_read_pointer = 0

        #print("Frames per second:", 1/(time.time() - start_t))
        self.end_of_run.emit() # emit the signal to keep the loop going

    def stop(self): # gets triggered by self.worker.stop()
        self.kill_signal = True # used to stop the RTL thread
        self.audio_stream.close()
        self.p.terminate() # Release PortAudio system resources
        print("Stopped audio stream")
        self.rtl_thread.join() # waits for RTL thread to end on its own
        print("RTL thread stopped")
