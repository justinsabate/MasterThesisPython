import numpy as np
import scipy
from librosa import load, resample
from sound_field_analysis import io, process
import sys

from sound_field_analysis.process import rfi

sys.path.insert(0, "/code_SH")
from code_SH.SphHarmUtils import *
import h5py
# needed for debugging
from sound_field_analysis import process
import soundfile as sf

# Initializations
start_time = 0
end_time = 3
signal_name = './wavfiles/BluesA_GitL.wav'
N = 4
sh_kind = "complex"
rotation_sound_field_deg = 120
output_file_name = 'output_old'
measurementFileName = 'database/DRIR anechoic Xenofon/SoundFieldControlRoomSphereData.hdf5'
hrtf_process = 1  # to speed up some debug not on the hrtf side
radial_plot = 1
amp_maxdB = 40  # for the radial filter
freq_output_plot = 1

'''Get the data'''
# Be sure they are the same type (float32 for ex)

### Measurement data
measurementFile = h5py.File(measurementFileName, "r")
# print(measurementFile.keys())
# print(list(measurementFile.attrs))
m = measurementFile['Measurements']  # float64 but should be ok with float32
# print(m.keys())
DRIR = m['RIRs']  # size 86 (mic) x 1487999 (samples),
# something like 15s of silence before the clap
fs_r = measurementFile.attrs['fs']  # Sampling frequency
g = measurementFile['Grid'] # grid, positions of the microphones
grid = g['received_grid_positions']  # size 3 (x,y,z) x 86 (microphone)

### HRTF set extraction
if hrtf_process:
    HRIR = io.read_SOFA_file("../database/HRIR TH Koln/HRIR_L2702.sofa")
    fs_h = int(HRIR.l.fs)
    NFFT = HRIR.l.signal.shape[-1]
else:
    NFFT = 128  # for debug

### Loading the anechoic signal to be convolved
s, fs_s = load(signal_name, sr=None, mono=True, duration=3, dtype=np.float32)

'''Data preparation'''
# set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono

# select shorter extract from source
s = s[int(start_time * fs_s):int(min(end_time * fs_s, np.shape(s)[0]))]

# same sampling frequency
if hrtf_process:
    fs_min = min([fs_r, fs_h, fs_s])
    if fs_r < fs_min:
        RIR = resample(DRIR, orig_sr=fs_r, target_sr=fs_min)
    elif fs_h < fs_min:
        HRIR = resample(HRIR, orig_sr=fs_h, target_sr=fs_min)  # works even if 2 channels
    elif fs_s < fs_min:
        s = resample(s, orig_sr=fs_s, target_sr=fs_min)
else:
    fs_min = 48000  # for debug


''' Preprocessing of the HRIR '''
# according to paper 11 : allpass reducing ITD at high frequencies



''' SH expansion '''

# HRTF
# transform SOFA data
if hrtf_process:
    HRTF_L = np.fft.fft(HRIR.l.signal, NFFT)[:, 0:int(NFFT / 2 + 1)]
    HRTF_R = np.fft.fft(HRIR.r.signal, NFFT)[:, 0:int(NFFT / 2 + 1)]

    Hnm = np.stack(
        [
            spatialFT(
                HRTF_L,
                HRIR.grid,
                grid_type='sphe',
                order_max=N,
                kind="complex",
                spherical_harmonic_bases=None,
                weight=None,
                leastsq_fit=True,
                regularised_lstsq_fit=False
            ),
            spatialFT(
                HRTF_R,
                HRIR.grid,
                grid_type='sphe',
                order_max=N,
                kind="complex",
                spherical_harmonic_bases=None,
                weight=None,
                leastsq_fit=True,
                regularised_lstsq_fit=False
            ),
        ]
    )

    # Debugs

    # Hnm = np.stack(
    #     [
    #         process.spatFT(
    #             process.FFT(HRIR.l.signal, fs=fs_min, NFFT=NFFT, calculate_freqs=False),
    #             position_grid=HRIR.grid,
    #             order_max=N,
    #             kind=sh_kind,
    #         ),
    #         process.spatFT(
    #             process.FFT(HRIR.r.signal, fs=fs_min, NFFT=NFFT, calculate_freqs=False),
    #             position_grid=HRIR.grid,
    #             order_max=N,
    #             kind=sh_kind,
    #         ),
    #     ]
    # )

# DRIR

DRFR = np.fft.fft(DRIR, NFFT)[:, 0:int(NFFT / 2 + 1)]  # taking only the first part of the FFT, not the reversed part

Pnm = spatialFT( #Pnm : Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    DRFR,
    grid,
    grid_type='cart',
    order_max=N,
    kind="complex",
    spherical_harmonic_bases=None,
    weight=None,
    leastsq_fit=True,
    regularised_lstsq_fit=False)

# Debugs
# Pnm2 = process.spatFT_LSF(
#     # process.FFT(DRIR, fs=fs_min, NFFT=NFFT, calculate_freqs=False), # is only computing rfft (real fft not complex, but exact tame result, so not coming from here)
#     DRFR,
#     position_grid=grid,
#     order_max=N,
#     kind="complex",
#     spherical_harmonic_bases=None
# )

# np.allclose(base, base2)  # yes if extracted base from spatFT functions
# np.allclose(Pnm, Pnm2)  # yes

# End debug

''' Radial filter + smoothing of the coefficients'''
# TODO(actually design the filter and refinements described in the papers)
# NFFT has an influence on the filter shape (low frequencies)
freq = np.arange(0, NFFT // 2 + 1) * (fs_min / (NFFT))
r = 0.084 / 2
c = 343
kr = 2 * np.pi * freq * r / c
nm_size = np.shape(Pnm)[0]

"""
dn is actually the inverse of the output of the weight function
here a refinement is made to avoid very high values due to weight close to zero
"""

temp = np.transpose(weights(N, kr, 'rigid'))
i = kr[kr == N]
temp[temp == 0] = 1e-12
max_i = np.min([i for i, n in enumerate(kr) if n > N])

# weighting of the radial filter
amp_max = 10 ** (amp_maxdB / 20)
limiting_factor = (
        2
        * amp_max
        / np.pi
        * np.abs(temp)
        * np.arctan(np.pi / (2 * amp_max * np.abs(temp)))
)
dn = limiting_factor / temp
dn[:, max_i:] = 1 / temp[:, max_i:] # do not limit the filter for higher frequencies

# plotting the radial filters
if radial_plot:
    fig, ax = plt.subplots()  # plots
    for i in range(0, len(dn)):
        ax.plot(freq, 20 * np.log10(abs(dn[i])), label='N = ' + str(i))
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlim(np.min(freq) + 1, np.max(freq))
    # ax.set_ylim(np.min(abs(dn[1])), np.max(abs(dn[1])))
    plt.legend()
    plt.show()



'''# SH domain convolution'''
# Compute for each frequency bin

# Orientation
alpha = np.deg2rad(rotation_sound_field_deg)

"""
Complex convolution : Has to take the opposite of the m but not the n cf paper [12] (justin's references)
"""
nm_reversed_m = reverse_nm(N)

Sl = np.zeros(np.shape(freq))
Sr = np.zeros(np.shape(freq))

for nm in range(0, nm_size):  # nm (order and which spherical harmonic function in the order)
    n, m = get_n_m(nm, N)  # N is the maximum order to reach for n
    am = (-1) ** m
    dn_am_Pnm_HnmL_exp, dn_am_Pnm_HnmR_exp = dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[0][nm] * np.exp(-1j * m * alpha), \
                                             dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[1][nm] * np.exp(-1j * m * alpha)
    Sl = np.add(Sl, dn_am_Pnm_HnmL_exp)
    Sr = np.add(Sr, dn_am_Pnm_HnmR_exp)

# plots for debug
if freq_output_plot:
    fig, ax = plt.subplots()
    ax.plot(freq, 10*np.log10(abs(Sl.real)),label='Left ear signal')
    ax.plot(freq, 10*np.log10(abs(Sr.real)),label='Right ear signal')
    ax.set_xscale('log')
    plt.legend()
    ax.set_ylim(-100, -50)
    plt.title('Frequency domain signal obtained after spherical convolution')
    plt.show()

''' iFFT '''
sl, sr = process.iFFT(Sl), process.iFFT(
    Sr)  # use np for more consistency in the code (would not need the toolbox in the end)

''' Convolution '''
sl_out, sr_out = np.transpose(scipy.signal.fftconvolve(sl[0], s)), np.transpose(scipy.signal.fftconvolve(sr[0], s))

''' Scaling '''
max = np.maximum(np.max(sl_out), np.max(sr_out))
sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''
sf.write('./exports/' + output_file_name + '.wav', np.stack((sl_out, sr_out), axis=1), fs_min)
