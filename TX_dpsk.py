import uhd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal as sp_signal
from rcosdesign_custom import rcosdesign_custom


# ============================================================================
# Initialize Simulation
# ============================================================================
len_msg = 10000  # Message Length
M = 8  # Modulation Index
OF = 2  # Oversampling Factor

header = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=int) # Örnek 10 bitlik imza
payload = np.random.randint(0, M, len_msg-len(header))  # Create random payload
msg = np.concatenate((header, payload))

debug_level = 0  # Plot details flag


msg_log = "message_tx.txt"
np.savetxt(msg_log, msg, fmt='%d')

center_freq = 2.4e9
sample_rate = 1e6
duration =  200*len_msg / sample_rate # seconds
gain =60 # [dB] start low then work your way up

# ============================================================================
# Differential PSK Modulation
# ============================================================================
phase_steps = msg * 2 * np.pi / M
tx_phase = np.zeros(len_msg)
tx_phase[0] = phase_steps[0]

for ind in range(1, len_msg):
    tx_phase[ind] = tx_phase[ind - 1] + phase_steps[ind]

psk_mod = np.cos(tx_phase) + 1j * np.sin(tx_phase)

# Oversampling PSK Modulation
psk_mod_os = np.zeros(len_msg * OF, dtype=complex)
for idx in range(0, len_msg * OF, OF):
    psk_mod_os[idx] = psk_mod[idx // OF]

# ============================================================================
# Pulse Shaping - Raised-Cosine Filter
# ============================================================================
beta = 0.4  # Rolloff factor
span = 10  # Filter span in symbols
sps = OF  # Samples per symbol

h = rcosdesign_custom(beta, span, sps, 'sqrt', 0)
psk_mod_shaped = sp_signal.convolve(psk_mod_os, h, mode='same')

samples = psk_mod_shaped.astype(np.complex64)
print(f"Prepared {len(samples)} samples for transmission.")
usrp = uhd.usrp.MultiUSRP("serial=307B5DF,type=b200")


while True: 
    usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
print(f"Sent {len(samples)} samples")

plt.plot(np.real(samples), label='Real Part')
plt.plot(np.imag(samples), label='Imaginary Part')
plt.title("Transmitted Signal")


# ========================================================================
# Constellation Diagram Summary (Real vs Imaginary)
# ========================================================================
fig = plt.figure(figsize=(16, 6))

ax1 = plt.subplot(2, 5, 1)
ax1.scatter(np.real(psk_mod), np.imag(psk_mod), alpha=0.5, s=10)
ax1.set_title("Modulated")
ax1.set_xlabel("Real")
ax1.set_ylabel("Imaginary")
ax1.grid(True)
ax1.axis('equal')

ax2 = plt.subplot(2, 5, 2)
ax2.scatter(np.real(psk_mod_os), np.imag(psk_mod_os), alpha=0.5, s=10)
ax2.set_title("Oversampled")
ax2.set_xlabel("Real")
ax2.set_ylabel("Imaginary")
ax2.grid(True)
ax2.axis('equal')

ax3 = plt.subplot(2, 5, 3)
ax3.scatter(np.real(psk_mod_shaped), np.imag(psk_mod_shaped), alpha=0.5, s=10)
ax3.set_title("Pulse Shaped")
ax3.set_xlabel("Real")
ax3.set_ylabel("Imaginary")
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.show()
