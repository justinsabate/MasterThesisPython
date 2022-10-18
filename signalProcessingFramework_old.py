import sys
import scipy
from librosa import load, resample
from sound_field_analysis import io  # to read HRTF set : sofa file
from code_SH.SphHarmUtils import *
import h5py
import soundfile as sf
sys.path.insert(0, "/Users/justinsabate/ThesisPython/code_SH")

# Initializations
N                           = 4

measurementFileName         = 'database/DRIR anechoic Xenofon/SoundFieldControlRoomSphereData.hdf5'

signal_name                 = './wavfiles/Flamenco1_U89.wav'
start_time                  = 0
end_time                    = 100

output_file_name            = 'flamenco'

radial_plot                 = 0
freq_output_plot            = 0

rotation_sound_field_deg    = 170
amp_maxdB                   = 18 # for the radial filter see ref [13] ch 3.6.6 and 3.7.3.3

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
g = measurementFile['Grid']  # grid, positions of the microphones
grid = g['received_grid_positions']  # size 3 (x,y,z) x 86 (microphone)

### HRTF set extraction

HRIR = io.read_SOFA_file("./database/HRIR TH Koln/HRIR_L2702.sofa")
fs_h = int(HRIR.l.fs)
NFFT = HRIR.l.signal.shape[-1]  # TODO(might be able to change this)
# NFFT = 1024

### Loading the anechoic signal to be convolved
s, fs_s = load(signal_name, sr=None, mono=True, offset=start_time, duration=end_time-start_time, dtype=np.float32)

'''Data preparation'''
# set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono

# same sampling frequency
fs_min = min([fs_r, fs_h, fs_s])
if fs_r < fs_min:
    RIR = resample(DRIR, orig_sr=fs_r, target_sr=fs_min)
elif fs_h < fs_min:
    HRIR = resample(HRIR, orig_sr=fs_h, target_sr=fs_min)  # works even if 2 channels
elif fs_s < fs_min:
    s = resample(s, orig_sr=fs_s, target_sr=fs_min)

''' Preprocessing of the HRIR '''
# according to paper 11 : allpass reducing ITD at high frequencies
# TODO(implement refinement)

''' SH expansion '''

# HRTF
HRTF_L = np.fft.rfft(HRIR.l.signal, NFFT)  # taking only the components [:, 0:int(NFFT / 2 + 1)] of the regular np.fft
HRTF_R = np.fft.rfft(HRIR.r.signal, NFFT)

Hnm = np.stack(  # 2 sides, L and R
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

# DRIR
DRFR = np.fft.fft(DRIR, NFFT)[:, 0:int(NFFT / 2 + 1)]  # taking only the first part of the FFT, not the reversed part

Pnm = spatialFT(  # Pnm : Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    DRFR,
    grid,
    grid_type='cart',
    order_max=N,
    kind="complex",
    spherical_harmonic_bases=None,
    weight=None,
    leastsq_fit=True,
    regularised_lstsq_fit=False
)

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

# weighting of the radial filter as described in ref [13]
amp_max = 10 ** (amp_maxdB / 20)
limiting_factor = (
        2
        * amp_max
        / np.pi
        * np.abs(temp)
        * np.arctan(np.pi / (2 * amp_max * np.abs(temp)))
)
dn = limiting_factor / temp
dn[:, max_i:] = 1 / temp[:, max_i:]  # do not limit the filter for higher frequencies



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
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Radial filters')
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
    ax.plot(freq, 10 * np.log10(abs(Sl)), label='Left ear signal')
    ax.plot(freq, 10 * np.log10(abs(Sr)), label='Right ear signal')
    ax.set_xscale('log')
    plt.legend()
    # ax.set_ylim(-100, -50)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Frequency domain signal obtained after spherical convolution')
    plt.show()

''' iFFT '''
sl, sr = np.fft.irfft(Sl, NFFT), np.fft.irfft(Sr,
                                              NFFT)  # using the complex conjugate reversed instead of doing it by hand and using np.fft.ifft

''' Convolution '''
sl_out, sr_out = np.transpose(scipy.signal.fftconvolve(sl, s)), np.transpose(scipy.signal.fftconvolve(sr, s))

''' Scaling / Amplification '''
max = np.maximum(np.max(sl_out), np.max(sr_out))
sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''
sf.write('./exports/' + output_file_name + ' rot=' + str(rotation_sound_field_deg) + ' limiting=' + str(amp_maxdB) + '.wav',
         np.stack((sl_out, sr_out), axis=1), fs_min)

"""
References
----------
[13] : Bernschütz, B. (2016). Microphone arrays and sound field decomposition for dynamic binaural recording. Technische Universitaet Berlin (Germany)
"""
