# python_sim

Signal processing simulations for QAM/DPSK transmission and reception. The project contains Python implementations of key synchronization and correction loops (timing, carrier recovery, and decision-directed loops) plus TX/RX demo scripts.

Note: This project is under construction and intended for educational purposes.

## Contents

- TX scripts: `TX_qam.py`, `TX_dpsk.py`
- RX scripts: `RX_qam.py`, `RX_dpsk.py`
- Core DSP blocks:
  - `coarse_frequency_correction.py`
  - `costas_loop.py`
  - `mthpower_loop.py`
  - `time_synchronizer.py`
  - `qam_fine_sync.py`
  - `qam_fine_sync_pll.py`
  - `qam_decision_directed_loop.py`
  - `rcosdesign_custom.py`
  - `psk_code.py`
  - `qam.py`

## Requirements

- Python 3.8+
- Common scientific stack (NumPy, SciPy, Matplotlib). If these are missing, install them with your preferred package manager.

## Quick start

Run a QAM transmit simulation:

- `python TX_qam.py`

Run a QAM receive simulation:

- `python RX_qam.py`

Run DPSK TX/RX:

- `python TX_dpsk.py`
- `python RX_dpsk.py`

## Demo outputs

QAM RX demo output (constellation and synchronization behavior):

![QAM demo output](extras/qam_demo.png)

DPSK RX demo output (phase tracking and symbol decisions):

![DPSK demo output](extras/dpsk_demo.png)

## Notes

- The scripts are intended to be executed from the project root folder.

## Troubleshooting

- If a script fails, verify your Python version and installed dependencies.
- If plots do not appear, ensure your Matplotlib backend is configured correctly.

