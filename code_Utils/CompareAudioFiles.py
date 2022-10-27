"""Python script aimed at comparing different wav files"""
from librosa import load
import numpy as np
import matplotlib.pyplot as plt

file1 = 'BluesA_GitL zeros in the beginning rot=0 pos=11 limiting=18NFFT=4096 realtime=0 HRTFmodif=1 Tapering=1 EQ=1'
file2 = 'BluesA_GitLno zeros in the beginning rot=0 pos=11 limiting=18NFFT=4096 realtime=0 HRTFmodif=1 Tapering=1 EQ=1'
extension = '.wav'

s1, fs1 = load('./exports/' + file1 + extension, sr=None, mono=True, dtype=np.float32)
s2, fs2 = load('./exports/' + file2 + extension, sr=None, mono=True, dtype=np.float32)

plt.plot(s1)
plt.plot(s2)
plt.show()

plt.plot(s1[600:]-s2[0:-600])
plt.show()