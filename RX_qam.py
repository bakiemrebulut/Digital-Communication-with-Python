import msvcrt
import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from costas_loop import costas_loop
from qam_fine_sync_pll import qam_fine_sync_pll
from rcosdesign_custom import rcosdesign_custom
from time_synchronizer import time_synchronizer
from coarse_frequency_correction import coarse_frequency_correction
from qam_decision_directed_loop import qam_decision_directed_loop
from qam_fine_sync import qam_fine_sync
# ============================================================================
# Initialize Simulation
# ============================================================================
M = 16  # Modulation Index
OF = 2  # Oversampling Factor
debug_level = 1  # flag for plot details 
# For qam

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


header = np.array([0, 15, 3, 12, 0, 15, 3, 12, 0, 15], dtype=int)
header_complex_for_corr = constellation[header]
len_header = len(header)
ber_array = []
chunk_size = 500

tx_signal = np.loadtxt("message_tx.txt", dtype=int)
len_msg = len(tx_signal)
print(f"Loaded TX signal of length {len_msg} symbols.")
num_samples = int(OF*len_msg*10)
center_freq = 2.4e9
sample_rate = 1e6
gain =60 # [dB] start low then work your way up


# ============================================================================
error_counter = 0
loop_counter = 0
usrp = uhd.usrp.MultiUSRP("serial=3273AEE,type=b200")
axes=None
if debug_level:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    plt.suptitle("Receiver Debugging Information")
    plt.ion()  # Turn on interactive mode
    plt.show()  # Clear the current figure
while True:
    loop_counter += 1

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
    # ===========================================================================
    # Coarse Frequency Correction
    # ============================================================================
    coarse_freq = coarse_frequency_correction(time_sync, sample_rate, 4, OF, 0)

    #pwr = np.mean(np.abs(coarse_freq)**2)
    #input_signal = coarse_freq / np.sqrt(pwr)
    fine_sync = qam_fine_sync_pll(coarse_freq, M, alpha=0.1, beta=0.005, gain_alpha=0.02, debug_level=debug_level, axes=axes   )

    # Header Ara (Fine Sync üzerinde)
    correlation = np.correlate(fine_sync, header_complex_for_corr, mode='valid')
    
    
    # Pikleri gücüne göre sırala (En güçlü ama geçerli olanı al)
    while True:
        best_peak_idx = np.argmax(np.abs(correlation)) 
        if np.abs(correlation[best_peak_idx]) < 2.0:
            print("Kilitlenemedi.")
            break # Sinyal yoksa zorlama
        potential_end = best_peak_idx + len_msg
        # 1. Kural: Paket buffer dışına taşmamalı
        if potential_end < len(fine_sync):
            # 2. Kural: En baştaki paketi atla ki PLL ısınsın (Preamble Etkisi)
            # Eğer peakin indeksi (çok baştaysa) ve başka alternatif varsa atla.
            if best_peak_idx < len_msg/2:
                correlation[best_peak_idx] = 0  # Geçersiz kıl
                continue
            #indexteki değer çok yüksekse (gürültü değilse) kabul et, yoksa sıradaki en iyiye bak
            if np.abs(correlation[best_peak_idx]) > 16.90:
                correlation[best_peak_idx] = 0
                continue                
            valid_peak_found = True
            break # İlk geçerli paketi bulduk, al ve çık.
        else:
            correlation[best_peak_idx] = 0  # Geçersiz kıl ve sıradaki en iyiye bak
            continue

    if not valid_peak_found:
        print("Geçerli paket bulunamadı (Hepsi sınır dışı).")
        error_counter += 1
        continue

    best_peak_idx = np.argmax(np.abs(correlation)) 
    # Kilitlenme kontrolü
    """if np.abs(correlation[best_peak_idx]) < 2.0: 
        error_counter += 1
        print("Kilitlenemedi")
        continue # Sinyal yoksa zorlama
    elif np.abs(correlation[best_peak_idx]) > 90.0:
        print("Çok güçlü sinyal, muhtemelen gürültü. Atlanıyor.")
        error_counter += 1
        continue"""
    
    start_idx = best_peak_idx
    end_idx = start_idx + len_msg
    """if end_idx > len(fine_sync): 
        print("Buffer yetmedi. end_idx:", end_idx, "len(fine_sync):", len(fine_sync))
        error_counter += 1
        continue    """
    # --- AÇI HESAPLAMA ---
    # Alınan Header (Baklava halindeki)
    rx_header_part = fine_sync[start_idx : start_idx + len_header]
    
    # Bu header, İdeal Header'a göre kaç derece dönük?
    # (Bu formül 45, 90, 13, -160... her açıyı bulur)
    # A = RX, B = Ref -> A * conj(B) açısı farkı verir.
    phase_diff_vec = rx_header_part * np.conj(header_complex_for_corr)
    rotation_error = np.angle(np.mean(phase_diff_vec))
    
    # --- DÜZELTME ---
    # Tüm paketi bu hata kadar tersine çevir
    raw_packet = fine_sync[start_idx : start_idx + len_msg]
    final_signal = raw_packet * np.exp(-1j * rotation_error)
  
    # Demodülasyon
    distances = np.abs(final_signal[:, np.newaxis] - constellation[np.newaxis, :])
    rx_aligned = np.argmin(distances, axis=1)
    rx_stream = []
    for i in range(0, len(rx_aligned), 500):
        chunk = rx_aligned[i+len_header : i+len_header + 500]
        rx_stream.extend(chunk)
        
    # ========================================================================
    # Constellation Diagram Summary (Real vs Imaginary)
    # ========================================================================
    if debug_level:
        
        axes[0,2].scatter(np.real(received_signal[-int(num_samples/2):]), np.imag(received_signal[-int(num_samples/2):]), alpha=0.5, s=1)
        axes[0,2].set_title("1 Received Signal")
        axes[0,2].set_xlabel("Real")
        axes[0,2].set_ylabel("Imaginary")
        axes[0,2].grid(True)
        axes[0,2].axis('equal')

        axes[1,2].scatter(np.real(matched[-int(num_samples/2):]), np.imag(matched[-int(num_samples/2):]), alpha=0.5, s=1)
        axes[1,2].scatter(np.real(time_sync[-int(num_samples/2):]), np.imag(time_sync[-int(num_samples/2):]), alpha=0.5, s=1,color="orange")
        axes[1,2].set_title("2 Matched + Time Sync")
        axes[1,2].set_xlabel("Real")
        axes[1,2].set_ylabel("Imaginary")
        axes[1,2].grid(True)
        axes[1,2].axis('equal')

        axes[2,2].scatter(np.real(coarse_freq[-int(num_samples/2):]), np.imag(coarse_freq[-int(num_samples/2):]), alpha=0.5, s=1)
        axes[2,2].set_title("3 Coarse Frequency Correction")
        axes[2,2].set_xlabel("Real")
        axes[2,2].set_ylabel("Imaginary")
        axes[2,2].grid(True)
        axes[2,2].axis('equal')

        axes[0,3].scatter(np.real(fine_sync[-int(num_samples/2):]), np.imag(fine_sync[-int(num_samples/2):]), alpha=0.5, s=1)
        axes[0,3].set_title("4 Fine Sync PLL")
        axes[0,3].set_xlabel("Real")
        axes[0,3].set_ylabel("Imaginary")
        axes[0,3].grid(True)
        axes[0,3].axis('equal')

        axes[1,3].scatter(np.real(final_signal[-int(num_samples/2):]), np.imag(final_signal[-int(num_samples/2):]), alpha=0.5, s=1)
        axes[1,3].set_title("5 Final Signal")
        axes[1,3].set_xlabel("Real")
        axes[1,3].set_ylabel("Imaginary")
        axes[1,3].grid(True)
        axes[1,3].axis('equal')


    # ============================================================================
    # Error Calculation
    # ============================================================================
    
    tx_aligned = tx_signal[:len(rx_aligned)]
    bit_errors = np.sum(rx_aligned != tx_aligned)
    ber_final = bit_errors / len(tx_aligned)
    print("correlation peak value:", np.abs(correlation[best_peak_idx]), f"\tBER: {ber_final:.6f}")
    if ber_final > 0.1:
        error_counter += 1
    #print(f"Final BER: {ber_final:.6f}")
    ber_array = np.append(ber_array, ber_final)
   
    # --- GÖRSEL KANIT (Debugging) ---
    if debug_level:
        """print("\n--- (First 20 Symbols) --------------")
        print(f"TX: {tx_aligned[:20]}")
        print(f"RX: {rx_aligned[:20]}")
        print("---------------------------------------")"""
        

        diff = tx_aligned != rx_aligned
        axes[2,3].plot(ber_array)
        axes[2,3].set_xlabel("Iteration")
        axes[2,3].set_ylabel("BER")
        axes[2,3].set_title(f"Bit Errors (AVG: {np.mean(ber_array):.5f})")
        plt.tight_layout()
        plt.pause(.2)  # Turn off interactive mode
    if msvcrt.kbhit():
            _ = msvcrt.getch()
            print("Key pressed. Exiting loop...")
            break

avg_ber = float(np.mean(ber_array)) 
error_rate = (error_counter / loop_counter)
print("\n=== Report ===")
print(f"Total loops: {loop_counter}")
print(f"Error count: {error_counter}")
print(f"Error rate: {error_counter}/{loop_counter} = {error_rate:.6f}")
print(f"Average BER: {avg_ber:.6f}")
if debug_level:
    plt.ioff()
    plt.show()


        

