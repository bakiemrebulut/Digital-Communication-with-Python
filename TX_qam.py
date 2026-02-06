import uhd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sp_signal
from rcosdesign_custom import rcosdesign_custom


# ============================================================================
# Initialize Simulation
# ============================================================================
len_msg = 5000  # Message Length
M = 16  # Modulation Index
OF = 2  # Oversampling Factor

header = np.array([0, 15, 3, 12, 0, 15, 3, 12, 0, 15], dtype=int)
payload = np.random.randint(0, M, len_msg - len(header))  # Create random payload
msg = np.concatenate((header, payload))
debug_level = 0  # Plot details flag


msg_log = "message_tx.txt"
np.savetxt(msg_log, msg, fmt='%d')

center_freq = 2.4e9
sample_rate = 1e6
duration =  200*len_msg / sample_rate # seconds
gain =60 # [dB] start low then work your way up

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

h = rcosdesign_custom(beta, span, sps, 'sqrt', 0)
qam_mod_shaped = sp_signal.convolve(qam_mod_os, h, mode='same')

samples = qam_mod_shaped.astype(np.complex64)
print(f"Prepared {len(samples)} samples for transmission.")
usrp = uhd.usrp.MultiUSRP("serial=307B5DF,type=b200")



while True: 
    usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
"""print(f"Sent {len(samples)} samples")

plt.plot(np.real(samples), label='Real Part')
plt.plot(np.imag(samples), label='Imaginary Part')
plt.title("Transmitted Signal")


# ========================================================================
# Constellation Diagram Summary (Real vs Imaginary)
# ========================================================================
fig = plt.figure(figsize=(16, 6))

ax1 = plt.subplot(2, 5, 1)
ax1.scatter(np.real(qam_mod), np.imag(qam_mod), alpha=0.5, s=10)
ax1.set_title("Modulated")
ax1.set_xlabel("Real")
ax1.set_ylabel("Imaginary")
ax1.grid(True)
ax1.axis('equal')

ax2 = plt.subplot(2, 5, 2)
ax2.scatter(np.real(qam_mod_os), np.imag(qam_mod_os), alpha=0.5, s=10)
ax2.set_title("Oversampled")
ax2.set_xlabel("Real")
ax2.set_ylabel("Imaginary")
ax2.grid(True)
ax2.axis('equal')

ax3 = plt.subplot(2, 5, 3)
ax3.scatter(np.real(qam_mod_shaped), np.imag(qam_mod_shaped), alpha=0.5, s=10)
ax3.set_title("Pulse Shaped")
ax3.set_xlabel("Real")
ax3.set_ylabel("Imaginary")
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.show()
"""