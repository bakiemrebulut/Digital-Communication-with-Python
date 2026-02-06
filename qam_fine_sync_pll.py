import numpy as np
import matplotlib.pyplot as plt 

def qam_fine_sync_pll(signal, M, alpha=0.05, beta=0.002, gain_alpha=0.01,debug_level=0, axes=None):
    """
    16-QAM için 2. Derece (Second Order) PLL ve AGC.
    Hem Frekans Ofsetini (Dönmeyi) hem Genliği düzeltir.
    
    alpha: Faz düzeltme hızı (Proportional Gain)
    beta:  Frekans takip hızı (Integral Gain) -> Dönmeyi durdurur
    gain_alpha: Genlik düzeltme hızı
    """
    N = len(signal)
    out = np.zeros(N, dtype=complex)
    
    phase_est = 0.0
    freq_est = 0.0  # Frekans ofseti hafızası (Integrator)
    gain_est = 1.0
    log= np.zeros(N)
    log2= np.zeros(N)
    log3= np.zeros(N)
    # 16-QAM Referans Tablosu (Normalize)
    if M == 16:
        ref_pts = np.array([-3, -1, 1, 3])
        I_ref, Q_ref = np.meshgrid(ref_pts, ref_pts)
        constellation = (I_ref.flatten() + 1j * Q_ref.flatten())
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
    
    phase_log = [] # Debug için
    
    for i in range(N):
        # 1. Düzeltme Uygula (PLL Mantığı: Faz + Frekans)
        # Sinyali; tahmini faz + birikmiş frekans ofseti kadar döndür
        sample = signal[i] * np.exp(-1j * phase_est) * gain_est
        
        # 2. Decision (En Yakın Nokta)
        dist = np.abs(sample - constellation)
        nearest_idx = np.argmin(dist)
        decision = constellation[nearest_idx]
        
        # 3. Hata Hesapla
        # Faz Hatası
        phase_err = np.angle(sample) -np.angle(decision) 
        phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi
        
        # Genlik Hatası
        gain_err = np.abs(decision) - np.abs(sample)
        
        # 4. Loop Filter (PI Controller)
        # Frekans Ofsetini Öğren (Integral kısmı)
        freq_est += beta * phase_err 
        
        # Fazı Güncelle (Proportional + Integral kısmı)
        phase_est += alpha * phase_err + freq_est
        
        # Genliği Güncelle
        gain_est += gain_alpha * gain_err
        
        out[i] = sample 
        phase_log.append(phase_est)
        if debug_level:
            log[i] = phase_est
            log2[i] = phase_err
            log3[i] = freq_est
    if debug_level == 1 and axes is not None:
        axes[0, 0].plot(log)
        axes[0, 0].set_title("Phase Estimate (PLL)")
        axes[1, 0].plot(log2)
        axes[1, 0].set_title("Phase Error (PLL)")
        axes[2, 0].plot(log3)
        axes[2, 0].set_title("Integrate (PLL)")
        plt.tight_layout()
    return out