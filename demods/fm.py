import numpy as np
from scipy.signal import firwin, bilinear, lfilter, filtfilt, lfilter_zi

class fm_demod():
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.h_lowpass = firwin(51, cutoff=50e3, fs=self.sample_rate)
        self.bz, self.az = bilinear(1, [75e-6, 1], fs=self.sample_rate) # De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
        self.zi = lfilter_zi(self.bz, self.az)
        
    def process(self, x):
        x = np.convolve(x, self.h_lowpass, 'same') # Low pass filter
        x = x[::4] # decimate, simply to zoom into the signal
        x = np.diff(np.unwrap(np.angle(x))) # FM demod
        x, self.zi = lfilter(self.bz, self.az, x, zi=self.zi) # apply de-emphasis filter
        x = x[::6] # decimate by 6 to get mono audio
        new_sample_rate = self.sample_rate/6/4
        #print("Sample rate audio:", sample_rate_audio)
        #h_audio = firwin(21, cutoff=2e3, fs=sample_rate_audio)
        #x = np.convolve(x, h_audio, 'same') # Low pass filter to get rid of popping
        return x, new_sample_rate


