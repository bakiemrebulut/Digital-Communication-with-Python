import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal


def mthpower_loop(signal_input, M, m, debug_level=0):
    """
    Mth-Power Loop for phase synchronization
    
    Parameters:
    - signal_input: input signal
    - M: modulation order
    - m: power order for the loop
    - debug_level: if 1, plots debugging information
    
    Returns:
    - phase_sync: phase synchronized signal
    - phase: final phase estimate
    """
    phase = 0
    integrate = 0
    
    # Set loop parameters based on modulation order
    if M == 2:
        Kp = 0.001
        Ki = 0.000001
    elif M == 4:
        Kp = 0.0025
        Ki = 0.0000025
    elif M >= 8:
        Kp = 0.003
        Ki = 0.000002
    
    N = len(signal_input)
    phase_log = np.zeros(N)
    phase_log2 = np.zeros(N)
    phase_log3 = np.zeros(N)
    error = 1
    
    for indx in range(N):
        out = signal_input[indx] * np.exp(-1j * phase)
        error = np.angle(out ** m) / m
        
        integrate = integrate + Ki * error
        phase = phase + integrate + Kp * error
        
        if debug_level:
            phase_log[indx] = phase
            phase_log2[indx] = error
            phase_log3[indx] = integrate
    
    phase_sync = signal_input * np.exp(-1j * phase)
    
    if debug_level == 1:
        fig, axes = plt.subplots(3, 1)
        axes[0].plot(phase_log)
        axes[0].set_title("phase_log")
        axes[1].plot(phase_log2)
        axes[1].set_title("error")
        axes[2].plot(phase_log3)
        axes[2].set_title("integrate")
        plt.tight_layout()
        plt.show()
    
    return phase_sync, phase
