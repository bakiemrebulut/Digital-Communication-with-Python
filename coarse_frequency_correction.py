import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def coarse_frequency_correction(time_sync, sample_rate, M,OF, debug_level=0):
    # 1. Modülasyonu kaldır (BPSK için karesini al)
    # Bu işlem BPSK verisini (0, 180 derece) tek bir saf sinüse (0 derece) çevirir.
    # Geriye sadece frekans hatasından kaynaklanan dönme kalır.
    signal_sq = time_sync ** M

    # 2. Faz değişim hızını hesapla (Differential Phase)
    # Ardışık örnekler arasındaki faz farkını bulur.
    # x[n] * conj(x[n-1]) işlemi faz farkını verir.
    phase_diff = np.angle(signal_sq[1:] * np.conj(signal_sq[:-1]))

    # 3. Ortalama faz değişimini bul
    # Gürültüyü bastırmak için ortalamasını alıyoruz.
    avg_phase_diff = np.mean(phase_diff)

    # 4. Frekans ofsetini hesapla
    # time_sync çıkışındaki örnekleme hızı (symbol rate)
    fs_symbol = sample_rate / OF 

    # Formül: (Delta_Phase) / (2 * pi * T_symbol * 2) 
    # Sondaki *2, sinyalin karesini aldığımız için frekansı 2 katına çıkardığımızdan dolayı bölmek içindir.
    freq_offset_est = avg_phase_diff / (2 * np.pi * (1/fs_symbol) * M)

    if debug_level:
        print(f"Hesaplanan Frekans Sapması (Offset): {freq_offset_est:.2f} Hz")

    # 5. Düzeltme Vektörünü Oluştur ve Uygula
    t_vec = np.arange(len(time_sync))
    correction_signal = np.exp(-1j * 2 * np.pi * freq_offset_est * t_vec / fs_symbol)

    time_sync_corrected = time_sync * correction_signal
    return time_sync_corrected