from fm import fm_demod
import numpy as np

def test_fm_demod():
    # Generate a simple FM signal
    num_samples = 100e3
    sample_rate = 1e6
    data_freq = 1e3
    fc = 100e3
    k = 0.05 # sensitivity
    
    t = np.arange(num_samples)/sample_rate
    signal = np.cos(2*np.pi*data_freq*t) # data signal
    x = np.exp(2j * np.pi * t * (fc + k*np.cumsum(signal))) # modulated FM signal, no carrier needed
    x = x.astype(complex)
    x += 0.1*np.random.randn(len(x)) + 0.1j*np.random.randn(len(x)) # Add some complex noise to avoid infs
    x = x * np.exp(-2j*np.pi*fc*t) # downconvert

    # plot PSD
    if False:
        import matplotlib.pyplot as plt
        PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2)
        f = np.linspace(-sample_rate/2.0, sample_rate/2.0, len(x))
        print(PSD)
        plt.plot(f/1e3, PSD)
        plt.xlabel("Frequency [kHz]")
        plt.ylabel("PSD")
        plt.show()

    # Demodulate the signal
    fmdemod = fm_demod(sample_rate)
    x, new_sample_rate = fmdemod.process(x)

    # plot PSD
    if False:
        import matplotlib.pyplot as plt
        PSD = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x)))**2)
        PSD = PSD[len(x)//2:] # real signal, only plot positive frequencies
        f = np.linspace(0, new_sample_rate/2.0, len(x)//2 + 1)
        plt.plot(f/1e3, PSD)
        plt.xlabel("Frequency [kHz]")
        plt.ylabel("PSD")
        plt.show()
    
    # Find freq of spike
    PSD = np.abs(np.fft.fft(x))**2
    PSD = PSD[0:len(x)//2] # real signal
    f = np.linspace(0, new_sample_rate/2.0, len(x)//2)
    max_f = f[np.argmax(PSD)]
    
    # Check that the spike is at the right freq
    assert np.abs(max_f - data_freq) < 5 # with 5 Hz

    # Check that the output is the right length, we shouldnt add or lose samples as part of the function
    assert np.abs(len(x) - num_samples*new_sample_rate/sample_rate) < 2

    # Check that the output is real
    assert np.all(np.isreal(x))
