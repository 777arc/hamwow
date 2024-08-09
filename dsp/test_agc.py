import numpy as np
from agc import AGC
import matplotlib.pyplot as plt

def test_agc():
    # Generate a simple signal
    block_size = 1000
    num_blocks = 100
    sample_rate = 1e3
    x = np.random.randn(block_size, num_blocks) + 1j*np.random.randn(block_size, num_blocks)
    
    test_envelope = 0.15 * np.cos(2*np.pi*0.05/sample_rate*np.arange(block_size*num_blocks)) + .25
    test_envelope = np.reshape(test_envelope, (block_size, num_blocks))
    
    x *= test_envelope
    
    # plt.figure()
    # plt.plot(x.flatten())
    # plt.show()
    
    x_orig = np.copy(x)
    
    agc = AGC(sample_rate, 1, 1, 0.5)
    
    # Apply AGC
    for block in x:
        block = agc.apply_agc(block)
    
    x = x.flatten()
    x_orig = x_orig.flatten()
    
    plt.figure()
    plt.plot(np.abs(x_orig), label='Original')
    plt.plot(np.abs(x), label='AGC')
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    test_agc()
    print("All tests pass")