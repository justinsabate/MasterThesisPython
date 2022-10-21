import sys

import numpy as np
import scipy
from librosa import load, resample
from sound_field_analysis import io, process, gen, utils  # to read HRTF set : sofa file
from sound_field_analysis.io import ArrayConfiguration

from code_SH.SphHarmUtils import *
import h5py
import soundfile as sf

from code_plots.plots_functions import plot_HRTF_refinement, plot_radial, plot_freq_output


sys.path.insert(0, "/Users/justinsabate/ThesisPython/code_SH")

# Initializations
N = 1  # 4 changed for MLS TODO(change this to 4 when inverse pb working)

measurementFileName = 'database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'

signal_name = 'BluesA_GitL'  # without file extension, in wavfiles folder
extension = '.wav'
start_time = 0
end_time = 10
real_time = 0
HRTF_refinement = 0  # from [11]
tapering_win = 1  # from soud field analysis toolbox + cf [36]
eq = 1  # from sound field analysis toolbox + cf [23], could probably be calculated from HRTF but calculated from a sphere (scattering calculations)

output_file_name = signal_name

radial_plot = 0
freq_output_plot = 0
HRTF_refinement_plot = 0
grid_plot = 0
DRIR_plot = 0

rotation_sound_field_deg = 0
amp_maxdB = 18  # for the radial filter see ref [13] ch 3.6.6 and 3.7.3.3
Nwin = 512  # for the real time processing, lower : artifacts, higher : possible delay
channel = 10  # channel of the measurement that is being used
sampling_frequency = 32000  # below, one can clearly hear the difference

'''Mandatory conditions'''
if HRTF_refinement == 0:
    HRTF_refinement_plot = 0

'''Get the data'''
# Be sure they are the same type (float32 for ex)

### Measurement data

with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIR = f[measurementgroupkey]['RIRs'][channel, :,
           :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency
    # NFFT = np.shape(DRIR)[-1] # too big
    f.close()

grid = get_eigenmike_grid(plot=False)

if grid_plot:
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(grid[0], grid[1], grid[2], color="green")
    # ax.set_xlabel('')
    # norm  = grid[0]**2+(grid[1]+0.9)**2+grid[2]**2

### HRTF set extraction

HRIR = io.read_SOFA_file("./database/HRIR TH Koln/HRIR_L2702.sofa")  # changed for MLS"./database/HRIR TH Koln/HRIR_FULL2DEG.sofa"
fs_h = int(HRIR.l.fs)
HRIR_l_signal = HRIR.l.signal
HRIR_r_signal = HRIR.r.signal

### Loading the anechoic signal to be convolved
s, fs_s = load('./wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

'''Data preparation'''
# set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono-
    # s = s[1]

fs_min = min([fs_r, fs_h, fs_s, sampling_frequency])

# studying the effect of the truncation at the start

'finding the IR that starts the earliest'
# indice = 0
# mini = np.min([index for index, item in enumerate(DRIR[0]) if item > 0.001])
# for i in range(1,len(DRIR)):
#     temp = np.min([index for index, item in enumerate(DRIR[i]) if item > 0.001])
#     if temp < mini:
#         mini = temp
#         indice = i
# print(mini, indice)
'channel 9 starts the earliest, around indice = 6656'

# same sampling frequency
if fs_r > fs_min:
    DRIR = resample(DRIR, orig_sr=fs_r, target_sr=fs_min)
    print('RIR resampled')
    if fs_h > fs_min:
        HRIR_l_signal = resample(HRIR_l_signal, orig_sr=fs_h, target_sr=fs_min)
        HRIR_r_signal = resample(HRIR_r_signal, orig_sr=fs_h, target_sr=fs_min)
        print('HRIR resampled')
        if fs_s > fs_min:
            s = resample(s, orig_sr=fs_s, target_sr=fs_min)
            print('signal resampled')
    elif fs_s > fs_min:
        s = resample(s, orig_sr=fs_s, target_sr=fs_min)
        print('signal resampled')
elif fs_h > fs_min:
    HRIR_l_signal = resample(HRIR_l_signal, orig_sr=fs_h, target_sr=fs_min)
    HRIR_r_signal = resample(HRIR_r_signal, orig_sr=fs_h, target_sr=fs_min)
    print(' HRIR resampled')
    if fs_s > fs_min:
        s = resample(s, orig_sr=fs_s, target_sr=fs_min)
        print('signal resampled')
elif fs_s > fs_min:
    s = resample(s, orig_sr=fs_s, target_sr=fs_min)
    print('signal resampled')
else:
    print('non resampled')


# Taking 50ms/100ms of the RIR signal
# tries : 6000 vs 6600 for the first index of the RIR (before resampling)
nstart = int(6400/fs_h*fs_min)
nend = int(nstart + 0.1 * fs_min)

DRIR = DRIR[:, nstart:nend]

# studying the effect of the truncation in the beginning
if DRIR_plot:
    plt.plot(DRIR[9])
    plt.show()

i = 0
while 2 ** i < nend - nstart:
    if i < 15:
        i += 1
    else:
        print('Look at the data to choose NFFT')
        break

NFFT = 2 ** i  # might be a cause for a very long lasting run of the code, that is why it's printed

NFFT = 4096
print('Nfft=' + str(NFFT))

# for plots
freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

'''Preprocessing of the HRIR, according to paper [11], allpass reducing ITD at high frequencies to code a higher 
order even with a lower order '''

# creating the filter
fc = 1500  # empirically chosen in [11]
c = 343
if HRTF_refinement:
    rh = 8.5e-2  # HRIR.grid.radius
    azi = HRIR.grid.azimuth  # already spherical
    elev = HRIR.grid.colatitude
    tau_r = -np.cos(azi) * np.sin(elev) * rh / c
    tau_l = -tau_r

    # NFFT = 2048
    # freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

    allPassFilter_r = np.ones((len(tau_r), NFFT // 2 + 1), dtype=np.complex128)  # shape : nb positions x frequency bins
    allPassFilter_l = np.ones((len(tau_l), NFFT // 2 + 1), dtype=np.complex128)

    for i, f in enumerate(freq):
        if f > fc:
            allPassFilter_r[:, i] = np.exp(
                -1j * 2 * np.pi * (f - fc) * tau_r)  # TODO(implement the filter to compensate precisely for the ITD)
            allPassFilter_l[:, i] = np.exp(-1j * 2 * np.pi * (f - fc) * tau_l)

# fft
HRTF_L = np.fft.rfft(HRIR_l_signal, NFFT)  # taking only the components [:, 0:int(NFFT / 2 + 1)] of the regular np.fft
HRTF_R = np.fft.rfft(HRIR_r_signal, NFFT)


# applying the filter
if HRTF_refinement:
    HRTF_L = HRTF_L * allPassFilter_l
    HRTF_R = HRTF_R * allPassFilter_r

# Plots of the filter magnitude and phase
plot_HRTF_refinement(freq, allPassFilter_l, allPassFilter_r, HRTF_L, HRTF_R, NFFT, HRTF_refinement_plot)

''' SH expansion '''

# HRTF done just above

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
            leastsq_fit=False,
            regularised_lstsq_fit=False,
            MLS=True
        ),
        spatialFT(
            HRTF_R,
            HRIR.grid,
            grid_type='sphe',
            order_max=N,
            kind="complex",
            spherical_harmonic_bases=None,
            weight=None,
            leastsq_fit=False,
            regularised_lstsq_fit=False,
            MLS=True
        ),
    ]
)


DRFR = np.fft.rfft(DRIR, NFFT) # rfft takes only the first part of the fft

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

"""
dn is actually the inverse of the output of the weight function
here a refinement is made to avoid very high values due to weight close to zero
"""
'Old code was taking into account the function called weights() but it cannot be done if the grid ' \
    'is not a grid that allows direct integration, so now the sofia toolbox (which gives similar but ' \
    'correct results) is used'

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

# make radial filters causal
dn_delay_samples = NFFT / 2
dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)
# additional treatment possible in the toolbox to remove the dc component with high passing, not made here

# improvement EQing the orders that are truncated
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

# improvement : tapering
if tapering_win:  # half sided hanning window
    w_tap = gen.tapering_window(N)
    dn = np.transpose(np.transpose(dn) * w_tap)

# plotting the radial filters, the phase has to be linear (causal filter)
plot_radial(freq, dn, radial_plot)

''' SH domain convolution'''

# Compute for each frequency bin
# Orientation
alpha = np.deg2rad(rotation_sound_field_deg)
print(alpha)

"""
Complex convolution : Has to take the opposite of the m but not the n cf paper [12] (justin's references)
"""

nm_reversed_m = reverse_nm(N)

Sl = np.zeros(np.shape(freq))
Sr = np.zeros(np.shape(freq))

for nm in range(0, nm_size):  # nm (order and which spherical harmonic function in the order)
    n, m = get_n_m(nm, N)  # N is the maximum order to reach for n
    am = (-1) ** m
    dn_am_Pnm_HnmL_exp = dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[0][nm] * np.exp(-1j * m * alpha)
    dn_am_Pnm_HnmR_exp = dn[n] * am * Pnm[nm_reversed_m[nm]] * Hnm[1][nm] * np.exp(-1j * m * alpha)
    Sl = np.add(Sl, dn_am_Pnm_HnmL_exp)
    Sr = np.add(Sr, dn_am_Pnm_HnmR_exp)

# plots for debug
plot_freq_output(freq, Sl, Sr, freq_output_plot)

''' iFFT '''

sl, sr = np.fft.irfft(Sl, NFFT), np.fft.irfft(Sr, NFFT)
# using the complex conjugate reversed instead of doing it by hand and using np.fft.ifft

''' Convolution '''

if not real_time:
    sl_out, sr_out = np.transpose(scipy.signal.fftconvolve(sl, s)), np.transpose(scipy.signal.fftconvolve(sr, s))
else:
    Nshift = Nwin // 2  # for overlap

    # zeropadd signal to match nb of windows
    if np.shape(s)[0] % Nwin:  # if not multiple of Nwin
        s = np.concatenate((s, np.zeros(Nwin - (np.shape(s)[0] % Nwin))), axis=0)
        print('signal zero padded')

    # zeropadd signal to add Nshift zeros in the end - FOR OVERLAP ADD
    s = np.concatenate((s, np.zeros(Nshift)), axis=0)

    # initialization
    s_sampled_size = (Nwin, np.shape(s)[0] // (Nwin - Nshift))
    s_sampled = np.zeros(s_sampled_size)
    sl_out_sampled = np.zeros(s_sampled_size)
    sr_out_sampled = np.zeros(s_sampled_size)
    sl_out = np.zeros(np.shape(s))
    sr_out = np.zeros(np.shape(s))

    # windowing because some processing will be done in the frequency domain
    win = np.sqrt(np.hanning(Nwin))

    # processing, mimic real time
    for n_sample in range(0, s_sampled_size[1] - 1):
        # sampling and windowing
        s_sampled[:, n_sample] = win * s[n_sample * Nshift:n_sample * Nshift + Nwin]
        # convolution
        sl_out_sampled[:, n_sample], sr_out_sampled[:, n_sample] = np.transpose(
            scipy.signal.fftconvolve(s_sampled[:, n_sample], sl, mode='same')), np.transpose(
            scipy.signal.fftconvolve(s_sampled[:, n_sample], sr, mode='same'))
        # overlap add
        sl_out[n_sample * Nshift:n_sample * Nshift + Nwin] += win * sl_out_sampled[:, n_sample]  # apply the second
        # window to have a smooth link between the samples
        sr_out[n_sample * Nshift:n_sample * Nshift + Nwin] += win * sr_out_sampled[:, n_sample]

    sl_out = sl_out[:-Nshift]  # to match the size of the input signal
    sr_out = sr_out[:-Nshift]

''' Scaling / Amplification '''
'The max value was calculated according to the highest amplitude in the output of the algorithm, for channel 11 of ' \
    'measurements DataEigenmikeDampedRoom10oct.hdf5, ' \
    'it depends on the resampling, usually if we lower the sampling frequency it has to go up'

max = np.maximum(np.max(np.abs(sl_out)), np.max(np.abs(sr_out)))
print(max)
# TODO(change this, comment above and uncomment below, this is just a try with other HRTF set+ other values for N)
# if fs_min == 32000:
#     max = 0.107 # sampling freq : 32 kHz
# elif fs_min == 48000:
#     max = 0.209  # sampling freq : 48 kHz
# else:
#     max = 0.209
# needs to be constant to be able to encode the distance (have different gains in different positions)

sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''

sf.write('./exports/{0} rot={1} pos={8} limiting={2}NFFT={3} realtime={4} HRTFmodif={5} Tapering={6} EQ={7}.wav'.format(
    output_file_name,
    str(rotation_sound_field_deg),
    str(amp_maxdB), str(NFFT),
    str(real_time),
    str(HRTF_refinement),
    str(tapering_win),
    str(eq),
    str(channel)),
    np.stack((sl_out, sr_out), axis=1), fs_min)

"""
References
----------
[11] : Zaunschirm, M., Schörkhuber, C., & Höldrich, R. (2018). Binaural rendering of Ambisonic signals by head-related impulse response time alignment and a diffuseness constraint. The Journal of the Acoustical Society of America, 143(6), 3616-3627
[13] : Bernschütz, B. (2016). Microphone arrays and sound field decomposition for dynamic binaural recording. Technische Universitaet Berlin (Germany)
"""
