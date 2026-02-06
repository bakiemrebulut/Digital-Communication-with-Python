import numpy as np
import matplotlib.pyplot as plt
def qam_fine_sync(signal, M, learning_rate_phase=0.01, learning_rate_gain=0.005, debug_level=0, axes=None):
    """
    16-QAM için Hem Faz (Phase) Hem Genlik (Gain) Takipçisi.
    Halkaları kare ızgaraya oturtur.
    """
    N = len(signal)
    out = np.zeros(N, dtype=complex)
    
    # Başlangıç değerleri
    phase_est = 0.0
    gain_est = 1.0
    log = np.zeros(N)
    log2 = np.zeros(N)
    log3 = np.zeros(N)
    # Referans Constellation (Normalize edilmiş)
    if M == 16:
        ref_pts = np.array([-3, -1, 1, 3])
        I_ref, Q_ref = np.meshgrid(ref_pts, ref_pts)
        constellation = (I_ref.flatten() + 1j * Q_ref.flatten())
        # Constellation'ı normalize et (Ortalama güç = 1)
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
    
    for i in range(N):
        # 1. Düzeltme Uygula (Hem Faz Hem Kazanç)
        # Sinyali önce döndür, sonra ölçekle
        val = signal[i] * np.exp(-1j * phase_est) * gain_est
        
        # 2. En Yakın Noktayı Bul (Decision)
        # Gelen 'val' değeri hangi ideal noktaya yakın?
        dist = np.abs(val - constellation)
        nearest_idx = np.argmin(dist)
        decision = constellation[nearest_idx]
        
        # 3. Hata Hesapla
        # Hata vektörü: İdeal nokta ile bizim noktamız arasındaki fark
        error_vec = decision - val
        
        # 4. Faz Hatası (Phase Error)
        # Basit açı farkı: angle(decision) - angle(val)
        # Küçük açılar için: Imag(val * conj(decision))
        phase_error = np.angle(decision) - np.angle(val)
        # Wrap phase (-pi, pi)
        phase_error = (phase_error + np.pi) % (2 * np.pi) - np.pi
        
        # 5. Genlik Hatası (Gain Error)
        # Bizim noktamız idealden büyük mü küçük mü?
        gain_error = np.abs(decision) - np.abs(val)
        
        # 6. Güncelle (LMS Update)
        phase_est += learning_rate_phase * phase_error
        gain_est += learning_rate_gain * gain_error
        if debug_level:
            log[i] = phase_est
            log2[i] = gain_est
            log3[i] = phase_error
        out[i] = val
    if debug_level == 1 and axes is not None:
        axes[0, 0].plot(log)
        axes[0, 0].set_title("Phase Estimate")
        axes[1, 0].plot(log2)
        axes[1, 0].set_title("Gain Estimate")
        axes[2, 0].plot(log3)
        axes[2, 0].set_title("Phase Error")
        plt.tight_layout()
    return out, phase_est