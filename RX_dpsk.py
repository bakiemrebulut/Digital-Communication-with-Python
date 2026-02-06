import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from costas_loop import costas_loop
from rcosdesign_custom import rcosdesign_custom
from time_synchronizer import time_synchronizer
from coarse_frequency_correction import coarse_frequency_correction
# ============================================================================
# Initialize Simulation
# ============================================================================
len_msg = 10000  # Message Length
M = 8  # Modulation Index
OF = 2  # Oversampling Factor
debug_level = 1  # flag for plot details 


num_samples = int(len_msg * OF)
center_freq = 2.4e9
sample_rate = 1e6
gain =60 # [dB] start low then work your way up

# For PSK
header = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=int)
header_complex_for_corr = np.exp(1j * header * 2 * np.pi / M)
ber_array = []
# ============================================================================

usrp = uhd.usrp.MultiUSRP("serial=3273AEE,type=b200")
axes=None
if debug_level:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    plt.suptitle("Receiver Debugging Information")
    plt.ion()  # Turn on interactive mode
    plt.show()  # Clear the current figure
while True:
    if debug_level:
        for ax in axes.flat:
            ax.cla()  # Clear each subplot
    samples = usrp.recv_num_samps(num_samples, center_freq, sample_rate, [0], gain) # units: N, Hz, Hz, list of channel IDs, dB
    samples = np.transpose(samples)
    #print(f"Received {len(samples)} samples")


    # ============================================================================
    # Matched Filter
    # ============================================================================
    beta = 0.4  # Rolloff factor
    span = 10  # Filter span in symbols
    sps = OF  # Samples per symbol

    h = rcosdesign_custom(beta, span, sps, 'sqrt', 0)
    received_signal = samples.flatten()
    matched = sp_signal.convolve(received_signal, h, mode='same')
    #print(received_signal[0])
    # ============================================================================
    # Time Synchronization
    # ============================================================================


    time_sync, tao = time_synchronizer(matched, OF, 'gardner', debug_level,axes)
    #print("Tao estimate:", tao )
    # ===========================================================================
    # Coarse Frequency Correction
    # ============================================================================
    coarse_freq = coarse_frequency_correction(time_sync, sample_rate, M, OF, debug_level)

    # ============================================================================
    # Costas Loop
    # ============================================================================
    phase_sync, phase = costas_loop(coarse_freq, M, debug_level, axes)

    # ============================================================================
    # Differential PSK Demodulation
    # ============================================================================
    rx_diff_complex = phase_sync[1:] * np.conj(phase_sync[:-1])
    rx_diff_angle = np.angle(np.concatenate([[0], rx_diff_complex]))
    rx_diff_angle[rx_diff_angle < 0] = rx_diff_angle[rx_diff_angle < 0] + 2 * np.pi

    demod_raw = np.round(rx_diff_angle * M / (2 * np.pi)).astype(int)
    recovps_sync = np.mod(demod_raw, M)
    log_file = "message_rx.txt"
    np.savetxt(log_file, recovps_sync, fmt='%d')
    # ========================================================================
    # Constellation Diagram Summary (Real vs Imaginary)
    # ========================================================================
    if debug_level:
        
        axes[0,2].scatter(np.real(received_signal[-int(num_samples/8):]), np.imag(received_signal[-int(num_samples/8):]), alpha=0.5, s=1)
        axes[0,2].set_title("1 Received Signal")
        axes[0,2].set_xlabel("Real")
        axes[0,2].set_ylabel("Imaginary")
        axes[0,2].grid(True)
        axes[0,2].axis('equal')

        axes[1,2].scatter(np.real(matched[-int(num_samples/8):]), np.imag(matched[-int(num_samples/8):]), alpha=0.5, s=1)
        axes[1,2].set_title("2 Matched")
        axes[1,2].set_xlabel("Real")
        axes[1,2].set_ylabel("Imaginary")
        axes[1,2].grid(True)
        axes[1,2].axis('equal')

        axes[2,2].scatter(np.real(time_sync[-int(num_samples/8):]), np.imag(time_sync[-int(num_samples/8):]), alpha=0.5, s=1)
        axes[2,2].set_title("3 Time Sync")
        axes[2,2].set_xlabel("Real")
        axes[2,2].set_ylabel("Imaginary")
        axes[2,2].grid(True)
        axes[2,2].axis('equal')

        axes[0,3].scatter(np.real(coarse_freq[-int(num_samples/8):]), np.imag(coarse_freq[-int(num_samples/8):]), alpha=0.5, s=1)
        axes[0,3].set_title("4 Coarse Frequency Correction")
        axes[0,3].set_xlabel("Real")
        axes[0,3].set_ylabel("Imaginary")
        axes[0,3].grid(True)
        axes[0,3].axis('equal')

        axes[1,3].scatter(np.real(phase_sync[-int(num_samples/8):]), np.imag(phase_sync[-int(num_samples/8):]), alpha=0.5, s=1)
        axes[1,3].set_title("5 Costas Out")
        axes[1,3].set_xlabel("Real")
        axes[1,3].set_ylabel("Imaginary")
        axes[1,3].grid(True)
        axes[1,3].axis('equal')


    # ============================================================================
    # Error Calculation
    # ============================================================================
    # ============================================================================
    # ROBUST FRAME SYNCHRONIZATION (Header Search)
    # ============================================================================
    
    
    rx_complex_for_corr = np.exp(1j * recovps_sync * 2 * np.pi / M)
    correlation = np.correlate(rx_complex_for_corr, header_complex_for_corr, mode='valid')
    
    # Mutlak değeri en yüksek olan noktayı bul (Faz dönmüş olsa bile yakalar)
    peak_index = np.argmax(np.abs(correlation))
    
    if debug_level: 
        print(f"Detected Start Index: {peak_index}")
        print(f"Correlation Strength: {np.abs(correlation[peak_index]):.2f} / {len(header)}")

    # 4. Hizalama ve Kesme
    # peak_index, header'ın başladığı yerdir.
    start_index = peak_index
    
    # Sinyalin sınırlarını belirle
    end_index = start_index + len_msg
    
    # Eğer kayıt erken bitmişse hata vermemesi için kontrol
    if end_index > len(recovps_sync):
        end_index = len(recovps_sync)
        
    rx_aligned = recovps_sync[start_index : end_index]
        
    # ============================================================================
    # Hata Hesaplama ve Görsel Kontrol
    # ============================================================================
    tx_signal = np.loadtxt("message_tx.txt", dtype=int)
    tx_aligned = tx_signal[:len(rx_aligned)]

    bit_errors = np.sum(rx_aligned != tx_aligned)
    ber_final = bit_errors / len(tx_aligned)
    
    print(f"Final BER: {ber_final:.6f}")
    
    # --- GÖRSEL KANIT (Debugging) ---
    if debug_level:
        print("\n--- (First 20 Symbols) --------------")
        print(f"TX: {tx_aligned[:20]}")
        print(f"RX: {rx_aligned[:20]}")
        print("---------------------------------------")
        ber_array = np.append(ber_array, ber_final)
        

        diff = tx_aligned != rx_aligned
        axes[2,3].plot(ber_array)
        axes[2,3].set_xlabel("Iteration")
        axes[2,3].set_ylabel("BER")
        axes[2,3].set_title(f"Bit Errors (AVG: {np.mean(ber_array):.5f})")
        axes[2,3].set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.pause(.1)  # Turn off interactive mode
        

