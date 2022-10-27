"""Python script aimed at preprocessing RIR to have different effects on the early reflections"""
import h5py
from librosa import load
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from code_Utils.MixTimeUtils import get_direct_index

'''If plain wav file'''
# file1 = 'BluesA_GitL zeros in the beginning rot=0 pos=11 limiting=18NFFT=4096 realtime=0 HRTFmodif=1 Tapering=1 EQ=1'
# extension = '.wav'
#
# s1, fs1 = load('./exports/' + file1 + extension, sr=None, mono=True, dtype=np.float32)


'''If h5py file, need to extract a position and a channel form this'''
measurementFileName = './database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
position = 4
channel = 9
outputFileName = 'processedIR'

with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIR = f[measurementgroupkey]['RIRs'][position, channel,
           :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency

    f.close()

'''Constants for plots'''
t = np.arange(0, len(DRIR))/fs_r

'''Create the window'''
Nwin = 512
win = np.hanning(Nwin)

indexDirect = get_direct_index(DRIR)[0] # to get only the first output
tDirect = indexDirect/fs_r
'''Apply the window '''
'Extracting the reflection(s)'
i_center = 7700
i_start = i_center-Nwin//2
i_end = i_center+Nwin//2
# t_mixing =  # should take the mixing time if multiple reflexions, adaptive window

refl_extractor = np.concatenate((np.zeros(i_start), win, np.zeros((len(DRIR)-i_end))))
refl = DRIR * refl_extractor

'Modification of the reflections'
gain = 0
refl *= gain

'Extracting the rest'
rest = DRIR * (1-refl_extractor)

'Reconstruction'
recon = rest + refl

'''Plot the window on the signal waveform'''
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(t, DRIR, label='Original RIR')

ax12 = ax1.twiny()
ax12.bar(tDirect,
        height=2*np.max(abs(DRIR)),
        width=0.001,
        tick_label='Direct sound: '+str(tDirect)[0:4]+'sec',
        bottom=-np.max(abs(DRIR)),
        color='red')
ax12.tick_params(axis='x', colors='red')

ax2.plot(t, refl, label='extracted reflection')
ax2.plot(t, rest, label='remaining RIR')

ax3.plot(t,refl_extractor, label='window')

ax4.plot(t, recon, label='processed RIR')

axs = fig.get_axes()
for ax in axs:
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid()
    ax.set_xlim((0.1, 0.3))
plt.show()


'''To export the processed RIR'''
# sf.write('./exports/{0} position {1} channel {2}.wav'.format(
#     outputFileName,
#     str(position),
#     str(channel)),
#     DRIR,
#     fs_r
# )
