import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from costas_loop import costas_loop
from rcosdesign_custom import rcosdesign_custom
from time_synchronizer import time_synchronizer


def symerr(msg_tx, msg_rx):
    """
    Calculate symbol errors
    
    Parameters:
    - msg_tx: transmitted message
    - msg_rx: received message
    
    Returns:
    - numerrs: number of errors
    - percentage_error: error percentage
    """
    errors = msg_tx != msg_rx
    numerrs = np.sum(errors)
    percentage_error = numerrs / len(msg_tx)
    return numerrs, percentage_error


def awgn(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to signal
    
    Parameters:
    - signal: input signal
    - snr_db: signal-to-noise ratio in dB
    
    Returns:
    - noisy_signal: signal with added noise
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise


# ============================================================================
# Initialize Simulation
# ============================================================================
len_msg = 40000  # Message Length
M = 16  # Modulation Index
OF = 2  # Oversampling Factor
plot_s = int(len_msg * 15 / 20)  # Plot start index
plot_d = OF * 40  # Plot window
debug_level = 1  # Plot details flag

msg = np.random.randint(0, M, len_msg)  # Create random message
awgn_SNR = 20

# ============================================================================
# QAM Modulation
# ============================================================================
if M == 16:
    LUT = [-3, -1, 1, 3]
elif M == 4:
    LUT = [-1, 1]
else:
    raise ValueError(f"M={M} not supported")

I, Q = np.meshgrid(LUT, LUT)
constellation = I.flatten() + 1j * Q.flatten()

# Power Normalization
avg_power = np.mean(np.abs(constellation) ** 2)
constellation = constellation / np.sqrt(avg_power)

qam_mod = constellation[msg]

# Oversampling QAM Modulation
qam_mod_os = np.zeros(len_msg * OF, dtype=complex)
for i in range(0, len_msg * OF, OF):
    qam_mod_os[i] = qam_mod[i // OF]

# ============================================================================
# Pulse Shaping - Raised-Cosine Filter
# ============================================================================
beta = 0.4  # Rolloff factor
span = 10  # Filter span in symbols
sps = OF  # Samples per symbol

h = rcosdesign_custom(beta, span, sps, 'sqrt', debug_level)
qam_mod_shaped = sp_signal.convolve(qam_mod_os, h, mode='same')

# ============================================================================
# Adding Time Delay
# ============================================================================
delay = np.random.randint(0, OF)
qam_mod_shaped_delay = np.zeros(len(qam_mod_shaped), dtype=complex)
if delay > 0:
    qam_mod_shaped_delay[delay:] = qam_mod_shaped[:-delay]
else:
    qam_mod_shaped_delay = qam_mod_shaped.copy()

# ============================================================================
# Phase Shifting
# ============================================================================
phase_shift = np.random.rand() * np.pi / 2 - np.pi / 4
qam_mod_shaped_delay_phase = qam_mod_shaped_delay * np.exp(1j * phase_shift)

# ============================================================================
# Adding AWGN Noise
# ============================================================================
received_signal = awgn(qam_mod_shaped_delay_phase, 20)

# ============================================================================
# Matched Filter
# ============================================================================
matched = sp_signal.convolve(received_signal, h, mode='same')

# ============================================================================
# Time Synchronization
# ============================================================================
time_sync, tao = time_synchronizer(matched, OF, 'gardner', debug_level)

# ============================================================================
# Costas Loop
# ============================================================================
phase_sync, phase = costas_loop(time_sync, M, debug_level)
# phase_sync, phase = mthpower_loop(time_sync, M, 4, 1)

# ============================================================================
# QAM Demodulation
# ============================================================================
# Power Normalization
avg_power2 = np.mean(np.abs(phase_sync) ** 2)
phase_sync_norm = phase_sync / np.sqrt(avg_power2)

# Find nearest constellation point
distances = np.abs(phase_sync_norm[:, np.newaxis] - constellation[np.newaxis, :])
recovered_sync = np.argmin(distances, axis=1)

# ============================================================================
# Error Calculation
# ============================================================================
numerrs_qamc, percentage_of_errorc = symerr(msg, recovered_sync)

if debug_level:
    plt.figure()
    plt.plot(msg - recovered_sync)
    plt.title("Error Signal")
    plt.show()

# ============================================================================
# Report
# ============================================================================
print(f'Modulation Level:\t\t{M}')
print(f'Oversampling Factor:\t{OF}\n')

print(f'Time Delay:\t\t\t\t{delay}')
print(f'Time Delay Estimation:\t{tao:.2f}\n')

print(f'Phase Shift:\t\t\t{phase_shift * 180 / np.pi:.2f} degree')
print(f'Phase Shift Estimation:\t{phase * 180 / np.pi:.2f} degree\n')
print(f'Phase Shift Error:\t\t{(phase_shift - phase) * 180 / np.pi:.2f} degree\n')

print(f'Number of Errors: {numerrs_qamc}')
print(f'Percentage of QAM Demodulation Error: {percentage_of_errorc * 100:.4f} %')

if debug_level:
    # ========================================================================
    # Synchronization vs Modulated Signal
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(np.arange(1, plot_d + 1), np.angle(received_signal[plot_s:plot_s + plot_d]))
    axes[0, 0].stem(np.arange(1, plot_d + 1), np.angle(qam_mod_os[plot_s:plot_s + plot_d]))
    axes[0, 0].legend(['angle of received', 'angle of modulated'])
    axes[0, 0].set_title("Received VS Modulated")
    
    axes[0, 1].plot(np.arange(1, plot_d + 1), np.angle(matched[plot_s:plot_s + plot_d]))
    axes[0, 1].stem(np.arange(1, plot_d + 1), np.angle(qam_mod_os[plot_s:plot_s + plot_d]))
    axes[0, 1].legend(['angle of matched', 'angle of modulated'])
    axes[0, 1].set_title("Matched VS Modulated")
    
    time_sync_slice = time_sync[plot_s // OF:plot_s // OF + plot_d // OF]
    qam_mod_slice = qam_mod[plot_s // OF:plot_s // OF + plot_d // OF]
    axes[1, 0].plot(np.arange(len(time_sync_slice)), np.angle(time_sync_slice))
    axes[1, 0].stem(np.arange(len(qam_mod_slice)), np.angle(qam_mod_slice))
    axes[1, 0].legend(['angle of time sync', 'angle of modulated'])
    axes[1, 0].set_title("Time Sync VS Modulated")
    
    phase_sync_slice = phase_sync[plot_s // OF:plot_s // OF + plot_d // OF]
    axes[1, 1].plot(np.arange(len(phase_sync_slice)), np.angle(phase_sync_slice))
    axes[1, 1].stem(np.arange(len(qam_mod_slice)), np.angle(qam_mod_slice))
    axes[1, 1].legend(['angle of costas', 'angle of modulated'])
    axes[1, 1].set_title("Costas vs Modulated")
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Pulse Shape
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    plot_range = slice(plot_s, plot_s + plot_d)
    
    axes[0, 0].plot(np.arange(plot_d), np.real(qam_mod_shaped[plot_range]))
    axes[0, 0].stem(np.arange(plot_d), np.real(qam_mod_os[plot_range]))
    axes[0, 0].legend(['real', 'real of modulated os'])
    axes[0, 0].set_title("Shaped vs Modulated")
    
    axes[0, 1].plot(np.arange(plot_d), np.imag(qam_mod_shaped[plot_range]))
    axes[0, 1].stem(np.arange(plot_d), np.imag(qam_mod_os[plot_range]))
    axes[0, 1].legend(['imag', 'imag of modulated os'])
    axes[0, 1].set_title("Shaped vs Modulated")
    
    axes[1, 0].plot(np.arange(plot_d), np.abs(qam_mod_shaped[plot_range]))
    axes[1, 0].stem(np.arange(plot_d), np.abs(qam_mod_os[plot_range]))
    axes[1, 0].legend(['abs', 'abs of modulated os'])
    axes[1, 0].set_title("Shaped vs Modulated")
    
    axes[1, 1].plot(np.arange(plot_d), np.angle(qam_mod_shaped[plot_range]))
    axes[1, 1].stem(np.arange(plot_d), np.angle(qam_mod_os[plot_range]))
    axes[1, 1].legend(['angle', 'angle of modulated os'])
    axes[1, 1].set_title("Shaped vs Modulated")
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Generated/Modulated/Demodulated
    # ========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    msg_slice = msg[plot_s // OF:plot_s // OF + plot_d // OF]
    qam_mod_slice = qam_mod[plot_s // OF:plot_s // OF + plot_d // OF]
    recovered_slice = recovered_sync[plot_s // OF:plot_s // OF + plot_d // OF]
    
    axes[0, 0].stem(np.arange(len(msg_slice)), msg_slice)
    axes[0, 0].set_title("Random Generated Bits")
    
    axes[0, 1].stem(np.arange(len(qam_mod_slice)), np.angle(qam_mod_slice))
    axes[0, 1].set_title("Phase of QAM Modulated")
    
    axes[1, 0].stem(np.arange(len(recovered_slice)), recovered_slice)
    axes[1, 0].set_title("Demodulated")
    
    axes[1, 1].stem(np.arange(len(qam_mod_slice)), np.abs(qam_mod_slice))
    axes[1, 1].set_title("Abs of Modulated and Oversampled")
    
    qam_mod_os_slice = qam_mod_os[plot_s:plot_s + (plot_d // OF - 1) * OF + 1]
    axes[2, 1].stem(np.arange(len(qam_mod_os_slice)), np.angle(qam_mod_os_slice))
    axes[2, 1].set_title("Phase of Modulated and Oversampled")
    
    axes[2, 0].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Channel Impact and Noise
    # ========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    channel_start = plot_s
    channel_end = plot_s + (plot_d - 1) * OF + 1
    channel_range = slice(channel_start, channel_end)
    
    channel_len = len(qam_mod_shaped[channel_range])
    
    axes[0].plot(np.arange(channel_len), np.real(qam_mod_shaped[channel_range]))
    axes[0].plot(np.arange(channel_len), np.real(qam_mod_shaped_delay[channel_range]))
    axes[0].stem(np.arange(channel_len), np.real(qam_mod_os[channel_range]))
    axes[0].legend(['Shaped', 'Delayed'])
    axes[0].set_title("Real Parts")
    
    axes[1].plot(np.arange(channel_len), np.imag(qam_mod_shaped[channel_range]))
    axes[1].plot(np.arange(channel_len), np.imag(qam_mod_shaped_delay[channel_range]))
    axes[1].stem(np.arange(channel_len), np.imag(qam_mod_os[channel_range]))
    axes[1].legend(['Shaped', 'Delayed'])
    axes[1].set_title("Imaginary Parts")
    
    axes[2].plot(np.arange(channel_len), np.abs(qam_mod_shaped_delay[channel_range]))
    axes[2].plot(np.arange(channel_len), np.abs(qam_mod_shaped_delay_phase[channel_range]))
    axes[2].plot(np.arange(channel_len), np.abs(received_signal[channel_range]))
    axes[2].stem(np.arange(channel_len), np.abs(qam_mod_os[channel_range]))
    axes[2].legend(['Delayed', 'Phase Shifted', 'Noise Added'])
    axes[2].set_title("Absolute")
    
    axes[3].plot(np.arange(channel_len), np.angle(qam_mod_shaped_delay[channel_range]))
    axes[3].plot(np.arange(channel_len), np.angle(qam_mod_shaped_delay_phase[channel_range]))
    axes[3].plot(np.arange(channel_len), np.angle(received_signal[channel_range]))
    axes[3].stem(np.arange(channel_len), np.angle(qam_mod_os[channel_range]))
    axes[3].legend(['Delayed', 'Phase Shifted', 'Noise Added'])
    axes[3].set_title("Angle")
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Constellation Diagram Summary (Real vs Imaginary)
    # ========================================================================
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = plt.subplot(2, 5, 1)
    ax1.scatter(np.real(qam_mod), np.imag(qam_mod), alpha=0.5, s=1)
    ax1.set_title("Modulated")
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.grid(True)
    ax1.axis('equal')
    
    ax2 = plt.subplot(2, 5, 2)
    ax2.scatter(np.real(qam_mod_os), np.imag(qam_mod_os), alpha=0.5, s=1)
    ax2.set_title("Oversampled")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imaginary")
    ax2.grid(True)
    ax2.axis('equal')
    
    ax3 = plt.subplot(2, 5, 3)
    ax3.scatter(np.real(qam_mod_shaped), np.imag(qam_mod_shaped), alpha=0.5, s=1)
    ax3.set_title("Pulse Shaped")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imaginary")
    ax3.grid(True)
    ax3.axis('equal')
    
    ax4 = plt.subplot(2, 5, 4)
    ax4.scatter(np.real(qam_mod_shaped_delay), np.imag(qam_mod_shaped_delay), alpha=0.5, s=1)
    ax4.set_title("Delayed")
    ax4.set_xlabel("Real")
    ax4.set_ylabel("Imaginary")
    ax4.grid(True)
    ax4.axis('equal')
    
    ax5 = plt.subplot(2, 5, 5)
    ax5.scatter(np.real(qam_mod_shaped_delay_phase), np.imag(qam_mod_shaped_delay_phase), alpha=0.5, s=1)
    ax5.set_title(f"Phase Shifted: {phase_shift*180/np.pi:.1f} deg")
    ax5.set_xlabel("Real")
    ax5.set_ylabel("Imaginary")
    ax5.grid(True)
    ax5.axis('equal')
    
    ax6 = plt.subplot(2, 5, 6)
    ax6.scatter(np.real(received_signal), np.imag(received_signal), alpha=0.5, s=1)
    ax6.set_title("Noise Added")
    ax6.set_xlabel("Real")
    ax6.set_ylabel("Imaginary")
    ax6.grid(True)
    ax6.axis('equal')
    
    ax7 = plt.subplot(2, 5, 7)
    ax7.scatter(np.real(matched), np.imag(matched), alpha=0.5, s=1)
    ax7.set_title("Matched")
    ax7.set_xlabel("Real")
    ax7.set_ylabel("Imaginary")
    ax7.grid(True)
    ax7.axis('equal')
    
    ax8 = plt.subplot(2, 5, 8)
    ax8.scatter(np.real(time_sync), np.imag(time_sync), alpha=0.5, s=1)
    ax8.set_title("Time Sync")
    ax8.set_xlabel("Real")
    ax8.set_ylabel("Imaginary")
    ax8.grid(True)
    ax8.axis('equal')
    
    ax9 = plt.subplot(2, 5, 9)
    ax9.scatter(np.real(phase_sync), np.imag(phase_sync), alpha=0.5, s=1)
    ax9.set_title("Costas Out")
    ax9.set_xlabel("Real")
    ax9.set_ylabel("Imaginary")
    ax9.grid(True)
    ax9.axis('equal')
    
    plt.tight_layout()
    plt.show()
