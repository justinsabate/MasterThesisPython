import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile
import librosa
import soundfile as sf
#########################
# Reading the wav files #
#########################
data, fs = librosa.load('./wavfiles/BluesA_GitL.wav', sr=None, mono=True,duration=3, dtype=np.float32)
HRIR, fsconvolve = librosa.load('./wavfiles/exHRIR2.wav', sr=None, mono=False, dtype=np.float32)

##############
# Resampling #
##############
# if fs > fsconvolve:
#     data = librosa.resample(data, fs, fsconvolve)
# elif fs < fsconvolve:
#     HRIR = librosa.resample(HRIR, fsconvolve, fs)

###################
# Defining x-axis #
###################
# x = [*np.linspace(0, len(data) / fs, num=len(data))]
# t = np.arange(0, (Nfft * 10)) / fs
# freq = [*np.arange(0, fs / 2, fs / Nfft)]

#####################
# One dimension fft #
#####################
# S = np.fft.fft(s, Nfft * 10)
# s2 = np.fft.ifft(S, Nfft * 10)

###############
# Convolution #
###############
# convolved1 = scipy.signal.fftconvolve(data, HRIR[0, :])
# convolved2 = scipy.signal.fftconvolve(data, HRIR[1, :])

##################
# Write wav file #
##################
# sf.write('exports/convolution.wav', np.transpose([convolved1,convolved2]), fs)

#################
# Tools to plot #
#################
# plt.plot(freq,abs(S[Nfft//2:]))
# plt.plot(t, s, label='s')
# plt.plot(t, s2, label='s2')
# plt.ylim([-32768, 32768])  # because wave file sometimes coded into int16
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude ')
# plt.grid()
#
# plt.show()

######################
# Plot in subfigures #
######################
# see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# fig, sub = plt.subplots(1,2)
# fig.suptitle('Convolved signals')
# sub[0].plot(convolved1)
# sub[1].plot(convolved2, 'tab:orange')
# sub[0].set_title('L')
# sub[1].set_title('R')
# plt.show()

############################
# Try to use SOFIA toolbox #
############################
# notebook accessible at this link : https://nbviewer.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/971cd1b62053afcdb4caa758d00af05affb48f86/examples/Exp4_BinauralRendering.ipynb

# read files from the hdf5 format
with h5py.File(filename, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    MetaDatagroupkey = list(f.keys())[1]
    datakeys = list(f[measurementgroupkey])
    metadatakeys = list(f[MetaDatagroupkey])
    attributes = list(f[MetaDatagroupkey].attrs.keys())
    print(f"{measurementgroupkey} group consists of: {datakeys}")
    print(f"{MetaDatagroupkey} group consists of: {metadatakeys} and {attributes}")
    rirs = f[measurementgroupkey]['RIRs'][:]
    temperatures = f[measurementgroupkey]['MeasuredTemperatures'][:]
    DOAs = f[measurementgroupkey]['LoudspeakerDOA'][:]
    distances = f[measurementgroupkey]['DistancesFromLoudspeaker'][:]
    bnoise = f[measurementgroupkey]['BackgroundNoise'][:]
    measured_fs = f[MetaDatagroupkey].attrs['fs']
    f.close()
