import numpy as np
from scipy.signal import firwin

class AGC:
    
    def __init__(self, sample_rate, attack_time, decay_time, target_level = 0.5):
        self.sample_rate = sample_rate
        self.attack_time = attack_time
        self.decay_time = decay_time
        self.target_level = target_level
        self.gain = 1.0
        
    def apply_agc(self, x):
        # Calculate the envelope of the signal
        envelope = np.abs(x)
        
        alpha = len(x) / self.sample_rate / self.attack_time if np.max(envelope) > self.target_level else \
            len(x) / self.sample_rate / self.decay_time
        
        # Update the gain based upon the desired time constants
        self.gain = self.gain ** (1 - alpha) * (self.target_level / np.max(envelope)) ** alpha
        
        x *= self.gain
        
        # print("Max envelope:", np.max(envelope))
        # print("Gain:", self.gain)
        
        return x
        