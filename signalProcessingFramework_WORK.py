import sys

import numpy as np
import scipy
from librosa import load, resample
from sound_field_analysis import io, process, gen, utils  # to read HRTF set : sofa file
from sound_field_analysis.io import ArrayConfiguration

from code_SH.SphHarmUtils import *
import h5py
import soundfile as sf

from code_Utils.MovSrcUtils import moving_convolution
from code_Utils.RealTimeUtils import overlap_add
from code_Utils.SamplingUtils import resample_if_needed
from code_plots.plots_functions import plot_HRTF_refinement, plot_radial, plot_freq_output, plot_grid

sys.path.insert(0, "/Users/justinsabate/ThesisPython/code_SH")

# Initializations
N = 1  # 1 to input for changed for MLS TODO(change this to 4 when inverse pb working)

measurementFileName = 'database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'

signal_name = 'BluesA_GitL'  # without file extension, in wavfiles folder
extension = '.wav'
start_time = 0
end_time = 10

position = 10  # position of the measurement that is being used

# rotation_sound_field_deg = np.arange(0, 360, 10)  # 0
rotation_sound_field_deg = [0]

real_time = 0
HRTF_refinement = 1  # from [11]
tapering_win = 1  # from soud field analysis toolbox + cf [36]
eq = 1  # from sound field analysis toolbox + cf [23], could probably be calculated from HRTF but calculated from a sphere (scattering calculations)

output_file_name = signal_name

radial_plot = 0
freq_output_plot = 0
HRTF_refinement_plot = 0
grid_plot = 0
DRIR_plot = 0

amp_maxdB = 18  # for the radial filter see ref [13] ch 3.6.6 and 3.7.3.3
Nwin = 512  # for the real time processing, but no time difference, might use it in the future
sampling_frequency = 32000  # below, one can clearly hear the difference

'''Mandatory conditions'''
if HRTF_refinement == 0:
    HRTF_refinement_plot = 0

if np.size(rotation_sound_field_deg) > 1:
    real_time = 0
    print('Real time set to 0, case with real time and many directions not implemented')

'''Get the data'''
# Be sure they are the same type (float32 for ex)

### Measurement data

with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIR = f[measurementgroupkey]['RIRs'][position, :,
           :]  # RIR, 106 measurements of the 32 positions eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency
    # NFFT = np.shape(DRIR)[-1] # too big
    f.close()

grid = get_eigenmike_grid(plot=False)
plot_grid(grid, grid_plot)

### HRTF set extraction

HRIR = io.read_SOFA_file(
    "./database/HRIR TH Koln/HRIR_L2702.sofa")  # changed for MLS"./database/HRIR TH Koln/HRIR_FULL2DEG.sofa"
fs_h = int(HRIR.l.fs)
HRIR_l_signal = HRIR.l.signal
HRIR_r_signal = HRIR.r.signal

### Loading the anechoic signal to be convolved
s, fs_s = load('./wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

'''Data preparation'''
### set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono-
    # s = s[1]

fs_min = min([fs_r, fs_h, fs_s, sampling_frequency])

### finding the IR that starts the earliest
# indice = 0
# mini = np.min([index for index, item in enumerate(DRIR[0]) if item > 0.001])
# for i in range(1,len(DRIR)):
#     temp = np.min([index for index, item in enumerate(DRIR[i]) if item > 0.001])
#     if temp < mini:
#         mini = temp
#         indice = i
# print(mini, indice)
'position 9 starts the earliest, around index = 6656'

### same sampling frequency
DRIR, HRIR_l_signal, HRIR_r_signal, s = resample_if_needed(fs_r, fs_min, fs_h, fs_s, DRIR, HRIR_l_signal, HRIR_r_signal,
                                                           s)

### Taking 50ms/100ms of the RIR signal
# tries : 6000 vs 6600 for the first index of the RIR (before resampling)
nstart = int(6400 / fs_h * fs_min)
nend = int(nstart + 0.1 * fs_min)

DRIR = DRIR[:, nstart:nend]

### studying the effect of the truncation in the beginning
if DRIR_plot:
    plt.plot(DRIR[9])
    plt.show()

### Choosing NFFT, important parameter
NFFT = 4096
print('Nfft=' + str(NFFT))

### abcissa for plots
freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

'''Preprocessing of the HRIR, according to paper [11], allpass reducing ITD at high frequencies to code a higher 
order even with a lower order '''

### creating the filter
fc = 1500  # empirically chosen in [11], consistent with duplex theory
c = 343
if HRTF_refinement:
    rh = 8.5e-2  # HRIR.grid.radius
    azi = HRIR.grid.azimuth  # already spherical
    elev = HRIR.grid.colatitude
    tau_r = -np.cos(azi) * np.sin(elev) * rh / c
    tau_l = -tau_r

    # NFFT = 2048
    # freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

    allPassFilter_r = np.ones((len(tau_r), NFFT // 2 + 1), dtype=np.complex128)  # shape : nb channels x frequency bins
    allPassFilter_l = np.ones((len(tau_l), NFFT // 2 + 1), dtype=np.complex128)

    for i, f in enumerate(freq):
        if f > fc:
            allPassFilter_r[:, i] = np.exp(
                -1j * 2 * np.pi * (f - fc) * tau_r)  # TODO(implement the filter to compensate precisely for the ITD)
            allPassFilter_l[:, i] = np.exp(-1j * 2 * np.pi * (f - fc) * tau_l)

### fft
HRTF_L = np.fft.rfft(HRIR_l_signal, NFFT)  # taking only the components [:, 0:int(NFFT / 2 + 1)] of the regular np.fft
HRTF_R = np.fft.rfft(HRIR_r_signal, NFFT)

### applying the filter
if HRTF_refinement:
    HRTF_L = HRTF_L * allPassFilter_l
    HRTF_R = HRTF_R * allPassFilter_r

    # Plots of the filter magnitude and phase
    plot_HRTF_refinement(freq, allPassFilter_l, allPassFilter_r, HRTF_L, HRTF_R, NFFT, HRTF_refinement_plot)

''' SH expansion '''

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
            regularised_lstsq_fit=False,
            MLS=False
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
            regularised_lstsq_fit=False,
            MLS=False
        ),
    ]
)

DRFR = np.fft.rfft(DRIR, NFFT)  # rfft takes only the first part of the fft

Pnm = spatialFT(  # Pnm : Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    DRFR,
    grid,
    grid_type='cart',
    order_max=N,
    kind="complex",
    spherical_harmonic_bases=None,
    weight=None,
    leastsq_fit=True,
    regularised_lstsq_fit=False,  # not nice result, don't know why
    MLS=False
)

''' Radial filter + smoothing of the coefficients'''

# NFFT has an influence on the filter shape (low frequencies)

r = 0.084 / 2
c = 343
kr = 2 * np.pi * freq * r / c
nm_size = np.shape(Pnm)[0]

### dn is actually the inverse of the output of the weight function here a refinement is made to avoid very high
# values due to weight close to zero


config = ArrayConfiguration(
    array_radius=0.085,
    array_type='rigid',
    transducer_type='omni',
    scatter_radius=0.085,
    dual_radius=None)

dn = gen.radial_filter_fullspec(
    max_order=N,
    NFFT=NFFT,
    fs=fs_min,
    array_configuration=config,
    amp_maxdB=amp_maxdB,
)

### make radial filters causal
dn_delay_samples = NFFT / 2
dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)
### additional treatment possible in the toolbox to remove the dc component with high passing, not made here

''' Implementing the different improvements of the auralized signal '''
### improvement EQing the orders that are truncated cf paper [23]
# takes into account tapering or not (adapts to it)
if eq:
    dn_shf = gen.spherical_head_filter_spec(
        max_order=N,
        NFFT=NFFT,
        fs=fs_min,
        radius=r,
        is_tapering=tapering_win,
    )

    # make Spherical Head Filter causal -> apparently no need because already causal

    dn_shf_delay_samples = NFFT / 2
    dn_shf *= gen.delay_fd(
        target_length_fd=dn_shf.shape[-1], delay_samples=dn_shf_delay_samples
    )

    dn[:] *= dn_shf  # effect : reversing the phase, and we can indeed hear a big difference, the phase is linear but
    # creates the delay necessary not to have weird "reflections" (with a hearable delay)

### improvement : tapering cf paper [23]
if tapering_win:  # half sided hanning window
    w_tap = gen.tapering_window(N)
    dn = np.transpose(np.transpose(dn) * w_tap)

### plotting the radial filters, the phase has to be linear (causal filter)
plot_radial(freq, dn, radial_plot)

''' SH domain convolution'''

# Compute for each frequency bin
# Orientation
alpha = np.array(np.deg2rad(rotation_sound_field_deg))  # to be able to call alpha[0] if
print('Head rotations (in grad) : ' + str(alpha))

###Complex convolution : Has to take the opposite of the m but not the n cf paper [12] (justin's references)
nm_reversed_m = reverse_nm(N)

### Initializations
Sl = np.zeros((np.size(alpha), len(freq)), dtype='complex')
Sr = np.zeros((np.size(alpha), len(freq)), dtype='complex')

### Calculating the frequency domain BRIR via SH convolution
for i in range(np.size(alpha)):  # if multiple directions to process, in case of moving source for example
    for nm in range(0, nm_size):  # nm (order and which spherical harmonic function in the order)
        n, m = get_n_m(nm, N)  # N is the maximum order to reach for n
        am = (-1) ** m
        dn_am_Pnm_HnmL_exp = dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[0][nm] * np.exp(-1j * m * alpha[i])
        dn_am_Pnm_HnmR_exp = dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[1][nm] * np.exp(-1j * m * alpha[i])
        Sl[i] = np.add(Sl[i], dn_am_Pnm_HnmL_exp)
        Sr[i] = np.add(Sr[i], dn_am_Pnm_HnmR_exp)

    # plots for debug
    plot_freq_output(freq, Sl, Sr, freq_output_plot)

''' iFFT '''

sl, sr = np.fft.irfft(Sl, NFFT), np.fft.irfft(Sr, NFFT)
# using the complex conjugate reversed instead of doing it by hand and using np.fft.ifft

''' Convolution with dry signal '''
speed = len(rotation_sound_field_deg) // 2  # this parameter encodes the speed of the change form one direction to
# another when the source is moving, if it is 0, then the source is not moving
if speed:  # moving source
    sl_out = moving_convolution(sl, s, speed)  # sl is of shape [nb of directions x signal size]
    sr_out = moving_convolution(sr, s, speed)
else:  # static source
    # remove 2 dimensionality
    sl = sl[0]
    sr = sr[0]
    if not real_time:
        sl_out, sr_out = np.transpose(scipy.signal.fftconvolve(sl, s)), np.transpose(scipy.signal.fftconvolve(sr, s))
    else:  # taking samples of the signal and overlapping the result (no effect for now)
        sl_out, sr_out = overlap_add(Nwin, s, sl, sr)

''' Scaling / Amplification '''
'The max value was calculated according to the highest amplitude in the output of the algorithm, for position 11 of ' \
'measurements DataEigenmikeDampedRoom10oct.hdf5, ' \
'it depends on the resampling, usually if we lower the sampling frequency it has to go up'

### For now if using MLS
# max = np.maximum(np.max(np.abs(sl_out)), np.max(np.abs(sr_out)))
# print(max)

# ## Regular use of it, measured on the position that leads to the biggest output and sets it (it is also
# sampling_frequency dependent

if fs_min == 32000:
    max = 0.107  # sampling freq : 32 kHz
elif fs_min == 48000:
    max = 0.209  # sampling freq : 48 kHz
else:
    max = 0.209
# needs to be constant to be able to encode the distance (have different gains in different positions)

sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''

sf.write('./exports/{0} pos={7} limiting={1} NFFT={2} realtime={3} HRTFmodif={4} Tapering={5} EQ={6}.wav'.format(
    output_file_name,
    str(amp_maxdB), str(NFFT),
    str(real_time),
    str(HRTF_refinement),
    str(tapering_win),
    str(eq),
    str(position)),
    np.stack((sl_out, sr_out), axis=1), fs_min)




"""
References
----------
[11] : Zaunschirm, M., Schörkhuber, C., & Höldrich, R. (2018). Binaural rendering of Ambisonic signals by head-related impulse response time alignment and a diffuseness constraint. The Journal of the Acoustical Society of America, 143(6), 3616-3627
[13] : Bernschütz, B. (2016). Microphone arrays and sound field decomposition for dynamic binaural recording. Technische Universitaet Berlin (Germany)
"""
