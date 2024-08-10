import numpy as np
from scipy.signal import firwin, bilinear, lfilter, filtfilt, lfilter_zi

class dsb_am():
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.h_lowpass = firwin(51, cutoff=50e3, fs=self.sample_rate)
        self.demod_lowpass = firwin(51, cutoff=3e3, fs=self.sample_rate/24)
        self.demod_highpass = firwin(51, cutoff=50, fs=self.sample_rate/24, pass_zero=False)
        
    def process(self, x):
        x = np.convolve(x, self.h_lowpass, 'same') # Low pass filter
        x = x[::24] # decimate, simply to zoom into the signal
        
        x = np.convolve(x, self.demod_lowpass, 'same') # Low pass filter
        x = np.convolve(x, self.demod_highpass, 'same') # High pass filter
        
        x = np.abs(x) # AM demod
        
        
        
        new_sample_rate = self.sample_rate/24
        #print("Sample rate audio:", sample_rate_audio)
        #h_audio = firwin(21, cutoff=2e3, fs=sample_rate_audio)
        #x = np.convolve(x, h_audio, 'same') # Low pass filter to get rid of popping
        return x, new_sample_rate