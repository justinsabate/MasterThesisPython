import sys
import h5py
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

measurementFileName = '../database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
position = 10
channel = 10
outputFileName = 'IR'

with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIR = f[measurementgroupkey]['RIRs'][position, channel,
           :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency

    f.close()

# DRIR = DRIR/np.max(DRIR)

# plt.plot(DRIR)
# plt.show()

# sf.write('test.wav',
#     np.array(DRIR),
#     fs_r
# )
sf.write('../exports/{0} position {1} channel {2}.wav'.format(
    outputFileName,
    str(position),
    str(channel)),
    DRIR,
    fs_r
)
