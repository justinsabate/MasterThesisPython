import sys
import h5py
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from librosa import load

from code_Utils.MixTimeUtils import data_based

'''Exporting one RIR from the spherical array measurement'''
measurementFileName = '../database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
position = 4
channel = [9, 10, 11]
outputFileName = 'IR_normalized'

with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIR = f[measurementgroupkey]['RIRs'][position, channel,
           :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency

    f.close()
#
# DRIR = DRIR/np.max(DRIR)
#
# sf.write('../exports/{0} position {1} channel {2}.wav'.format(
#     outputFileName,
#     str(position),
#     str(channel)),
#     DRIR,
#     fs_r
# )

#
'''Normalizing a stereo file'''
signal_name = '../external_code/matlab_code/mixing time paper 56/estimate_perceptual_mixing_time/example_data/1_EStud_220_0.4_BRIR_P00_P000.wav'
s, fs_s = load(signal_name, sr=None, mono=False)
# s = s/np.max([np.max(s[0,:]),np.max(s[1,:])])
# sf.write('./exports/normalized_test1.wav',
#     np.transpose(s),
#     fs_s
# )

## test measure mixing time
tmp50, tmp95, tmp50_interchannel_mean, tmp95_interchannel_mean, echo_dens = data_based(np.transpose(DRIR), fs_r)
print(tmp50)