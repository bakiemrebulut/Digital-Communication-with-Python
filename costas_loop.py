import numpy as np
import matplotlib.pyplot as plt

def get_loop_constants(loop_bw, damping_factor=0.707):
    """
    GNU Radio tarzı Alpha/Beta hesaplama.
    loop_bw: Genelde 0.01 ile 0.1 arasında seçilir (Normalized loop bandwidth)
    """
    denominator = (1.0 + 2.0 * damping_factor * loop_bw + loop_bw * loop_bw)
    alpha = (4 * damping_factor * loop_bw) / denominator
    beta = (4 * loop_bw * loop_bw) / denominator
    return alpha, beta
def costas_loop(signal, M, debug_level=0, axes=None):
    """
    Costas Loop for phase synchronization
    
    Parameters:
    - signal: input signal
    - M: modulation order (2, 4, 8, 16)
    - debug_level: if 1, plots debugging information
    
    Returns:
    - phase_sync: phase synchronized signal
    - phase: final phase estimate
    """
    phase = 0
    costas_integrate = 0
    freq_limit = 0.5 # Normalized frequency limit
 
    # Set loop parameters based on modulation order
    if M == 2:
        costas_Kp = 0.4
        costas_Ki = 0.0001
    elif M == 4:
        costas_Kp = 0.0025
        costas_Ki = 0.0000025
    elif M >= 8:
        costas_Kp = 0.008
        costas_Ki = 0.000002
    
    N = len(signal)
    costas_out = np.zeros(N, dtype=complex)
    phase_log = np.zeros(N)
    phase_log2 = np.zeros(N)
    phase_log3 = np.zeros(N)
    error = 1
    
    for i_costas in range(N):
        costas_out[i_costas] = signal[i_costas] * np.exp(-1j * phase)
        
        if M == 2:
            # Costas loop for BPSK
            error = np.sign(np.real(costas_out[i_costas])) * np.imag(costas_out[i_costas]) #np.real(costas_out[i_costas]) * np.imag(costas_out[i_costas])
        elif M == 4 or M == 16:  # QAM
            error = (np.sign(np.real(costas_out[i_costas])) * np.imag(costas_out[i_costas])) - \
                    (np.sign(np.imag(costas_out[i_costas])) * np.real(costas_out[i_costas]))
        elif M == 8:
            angle_costas = np.angle(costas_out[i_costas])
            ideal_theta = np.round(angle_costas * M / (2 * np.pi)) * (2 * np.pi / M)
            error = angle_costas - ideal_theta
        
        costas_integrate = costas_integrate + costas_Ki * error
        costas_integrate = np.clip(costas_integrate, -freq_limit, freq_limit)
        phase = phase + costas_integrate + costas_Kp * error
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        if debug_level and axes is not None:
            phase_log[i_costas] = phase
            phase_log2[i_costas] = error
            phase_log3[i_costas] = costas_integrate
    
    #phase_sync = signal * np.exp(-1j * phase)
    
    if debug_level == 1 and axes is not None:
        axes[0, 0].plot(phase_log)
        axes[0, 0].set_title("Phase Estimate")
        axes[1, 0].plot(phase_log2)
        axes[1, 0].set_title("Error")
        axes[2, 0].plot(phase_log3)
        axes[2, 0].set_title("Integrate")
        plt.tight_layout()
    
    return costas_out, phase
