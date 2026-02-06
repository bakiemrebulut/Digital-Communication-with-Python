import numpy as np
import matplotlib.pyplot as plt

def qam_decision_directed_loop(signal, constellation, learning_rate=0.05,debug_level=0,axes=None):
    """
    16-QAM için özel Faz Takip Döngüsü (Fine Phase Tracking).
    Sinyali en yakın constellation noktasına göre döndürür.
    """
    N = len(signal)
    out = np.zeros(N, dtype=complex)
    phase_est = 0.0
    phase_log = np.zeros(N)
    phase_log2 = np.zeros(N)
    phase_log3 = np.zeros(N)
    # Constellation referanslarını array olarak hazırla
    const_arr = np.array(constellation)
    
    for i in range(N):
        # 1. Mevcut faz tahmini ile sinyali düzelt
        sample = signal[i] * np.exp(-1j * phase_est)
        
        # 2. En yakın ideal noktayı bul (Decision)
        # (Bu işlem döngü içinde yavaş olabilir ama Python için en anlaşılır yol budur)
        dist = np.abs(sample - const_arr)
        nearest_idx = np.argmin(dist)
        decision = const_arr[nearest_idx]
        
        # 3. Hata Hesapla (Açı farkı)
        # Gelen sinyal ile karar verilen nokta arasındaki açı farkı
        error = np.angle(sample) - np.angle(decision)
        
        # Hata sarmalaması (Wrap -pi to pi)
        error = (error + np.pi) % (2 * np.pi) - np.pi
        
        # 4. Fazı güncelle (LMS Update)
        phase_est += learning_rate * error
        if debug_level:
            phase_log[i] = phase_est
            phase_log2[i] = error
            phase_log3[i] = learning_rate * error
        # Çıktıyı kaydet
        out[i] = sample
    if debug_level == 1 and axes is not None:
        axes[0, 0].plot(phase_log)
        axes[0, 0].set_title("Phase Estimate (QAM)")
        axes[1, 0].plot(phase_log2)
        axes[1, 0].set_title("Error (QAM)")
        axes[2, 0].plot(phase_log3)
        axes[2, 0].set_title("Phase Update (QAM)")
        plt.tight_layout()
        
    return out