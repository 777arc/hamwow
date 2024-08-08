import numpy as np
from scipy.signal import firwin, bilinear, lfilter

def fm_demod(x, sample_rate):
    h_lowpass = firwin(51, cutoff=50e3, fs=sample_rate)
    x = np.convolve(x, h_lowpass, 'same') # Low pass filter
    x = x[::4] # decimate, simply to zoom into the signal
    x = np.diff(np.unwrap(np.angle(x))) # FM demod
    bz, az = bilinear(1, [75e-6, 1], fs=sample_rate) # De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
    x = lfilter(bz, az, x) # apply de-emphasis filter
    x = x[::6] # decimate by 6 to get mono audio
    #sample_rate_audio = sample_rate/6/4
    #print("Sample rate audio:", sample_rate_audio)
    #h_audio = firwin(21, cutoff=2e3, fs=sample_rate_audio)
    #x = np.convolve(x, h_audio, 'same') # Low pass filter to get rid of popping
    return x


if __name__ == "__main__":
    # TODO - make a test that opens an example file and runs fm_demod and plots stuff
    pass

