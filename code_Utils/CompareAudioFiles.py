"""Python script aimed at comparing different wav files"""
from librosa import load
import numpy as np
import matplotlib.pyplot as plt
from soundfile import write

file1 = 'BluesA_GitL_no_processing'
file2 = 'BluesA_GitL pos=15 MLS=1 rfi=0 preprocessed=0'
extension = '.wav'

s1, fs1 = load('./exports/' + file1 + extension, sr=None, mono=False, dtype=np.float32)
s2, fs2 = load('./exports/' + file2 + extension, sr=None, mono=False, dtype=np.float32)


# plt.figure()
# plt.plot(s1[1])
# plt.plot(s2[1])
# plt.show()


'''Extension of the code to try to set the same loudness to compare the rest'''

prms1 = np.sqrt(np.sum(np.sum(s1, axis=0)**2)/np.shape(s1)[1])
prms2 = np.sqrt(np.sum(np.sum(s2, axis=0)**2)/np.shape(s2)[1])

s2 = s2*prms1/prms2  # normalization of the root mean square pressure(summed over the 2 chanels)

# plt.figure()
# plt.plot(s1[1])
# plt.plot(s2[1])
# plt.show()

write('exports/'+file2+'_normalised'+extension,s2.T,fs2)
write('exports/'+file1+'_normalised'+extension,s1.T,fs1)