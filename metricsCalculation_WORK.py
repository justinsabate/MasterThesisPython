"""File to implement the objective metrics rating the quality of the binauralized signals obtained with the file
signalProcessingFramework """
from librosa import load
import numpy as np
import matplotlib.pyplot as plt

from code_Utils.MetricsUtils import CLLerror, IC, ITD, ILD
from code_Utils.SamplingUtils import resample_if_needed


# Binaural signal
signal_name = 'BRIR pos=15 MLS=1 rfi=0 preprocessed=1'  # without file extension, in wavfiles folder
signal_extension = '.wav'

# Binaural reference
ref_name = 'BRIR pos=15 MLS=1 rfi=0 preprocessed=0'  # without file extension, in wavfiles folder
ref_extension = '.wav'

# Type of preprocessing - for plots
filtertype = 'gain'
title = 'Differences between the Binaural signals with and without preprocessing ('+filtertype+')'

# Length of the signals
start_time = 0
end_time = 10

# Loading the signals
s, fs_s = load('./exports/BRIR/' + signal_name + signal_extension, sr=None, mono=False, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

s_ref, fs_ref = load('./exports/BRIR/' + ref_name + ref_extension, sr=None, mono=False, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

# Resampling if needed
fs_min = min(fs_s, fs_ref)
resample_if_needed(fs_min, fs_ref, s_ref, fs_s, s)

# Calculation of CLL error
NFFT = 2048
error = CLLerror(s, s_ref, NFFT)  # one value per frequency

# Calculation of IC
ic_ref = IC(s_ref, fs_min, NFFT)
ic_s = IC(s, fs_min, NFFT)

# Calculation of ITD
itd = ITD(s, fs_min, plot=False)
itd_ref = ITD(s_ref, fs_min, plot=False)

# Calculation of ILD
fc, ild = ILD(s, fs_min)
fc, ild_ref = ILD(s_ref, fs_min)


# Initializations for plots
f = np.linspace(0, fs_min//2, NFFT//2+1)

# Plots
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(f, error, color='#1f77b4')
ax1.set_xscale('log')
# ax1.set_ylim(-40, 5)
# ax1.set_ylim(-175, 5)
# ax1.set_xlim(20, 0.5 * fs_min)
ax1.grid(True)
ax1.set_xlabel('Frequency (Hz)')

ax1.set_ylabel('CLL error (dB)') #, color='#1f77b4') #, fontsize='12'

# ax1.tick_params(axis='y', colors='#1f77b4') #, labelsize='12'

# IC

ax2 = fig.add_subplot(212)
ax2.plot(f, ic_ref, color='#1f77b4', label='reference')
ax2.plot(f, ic_s, color='#ff7f0e', label='processed BRIR')
ax2.set_xscale('log')
ax2.set_ylim(-1.1, 1.1)
# ax1.set_xlim(20, 0.5 * fs_min)
ax2.grid(True)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Interaural Coherence') #, fontsize='12'
# ax2.set_title(title)
# ax1.tick_params(axis='y', colors='#1f77b4') #, labelsize='12'

fig.suptitle(title)
plt.legend()
plt.draw()


# ILD
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(fc, ild, color='#1f77b4', label='processed BRIR')
ax1.plot(fc, ild_ref, color='#ff7f0e', label='reference BRIR')
ax1.set_xscale('log')
# ax1.set_ylim(-40, 5)
# ax1.set_ylim(-175, 5)
# ax1.set_xlim(20, 0.5 * fs_min)
ax1.grid(True)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('ILD (dB)')
ax1.legend()
fig.suptitle('ILD comparison')
plt.show()

