import sys

import numpy as np
import scipy
from librosa import load, resample
from sound_field_analysis import io, process, gen, utils  # to read HRTF set : sofa file
from sound_field_analysis.io import ArrayConfiguration

from code_SH.SphHarmUtils import *
import h5py
import soundfile as sf

sys.path.insert(0, "/Users/justinsabate/ThesisPython/code_SH")

# Initializations
N = 4

measurementFileName = 'database/DRIR anechoic Xenofon/AnechoicDataEigenmike.hdf5'

signal_name = 'badguy'  # without file extension, in wavfiles folder
extension = '.wav'
start_time = 0
end_time = 60
real_time = 0
HRTF_refinement = 1 # from [11]
tapering_win = 1  # from soud field analysis toolbox + cf [36]
eq = 1  # from sound field analysis toolbox + cf [23], could probably be calculated from HRTF but calculated from a sphere (scattering calculations)

output_file_name = signal_name+'right'

radial_plot = 0
freq_output_plot = 0
HRTF_refinement_plot = 0

rotation_sound_field_deg = 60
amp_maxdB = 18  # for the radial filter see ref [13] ch 3.6.6 and 3.7.3.3
Nwin = 512  # for the real time processing, lower : artifacts, higher : possible delay
channel = 0  # channel of the measurement that is being used

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

# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
#
# # Creating plot
# ax.scatter3D(grid[0], grid[1], grid[2], color="green")
# # ax.set_xlabel('')
# norm  = grid[0]**2+(grid[1]+0.9)**2+grid[2]**2

# ax.scatter3D(grid2[0], grid2[1], grid2[2], color="green")


### HRTF set extraction

HRIR = io.read_SOFA_file("./database/HRIR TH Koln/HRIR_FULL2DEG.sofa")
fs_h = int(HRIR.l.fs)

### Loading the anechoic signal to be convolved
s, fs_s = load('./wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

'''Data preparation'''
# set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono-
    # s = s[1]

fs_min = min([fs_r, fs_h, fs_s])

# Taking 100ms of the RIR signal
nstart = 49000
nend = int(nstart + 0.05 * fs_min)
DRIR = DRIR[:, nstart:nend]

i = 0
while 2 ** i < nend - nstart:
    if i < 15:
        i += 1
    else:
        print('Look at the data to choose NFFT')
        break

NFFT = 2 ** i
# difference with sofia
# NFFT = 25192
print('Nfft=' + str(NFFT))

# same sampling frequency
if fs_r < fs_min:
    RIR = resample(DRIR, orig_sr=fs_r, target_sr=fs_min)
    print('resampled')
elif fs_h < fs_min:
    HRIR.l.signal = resample(HRIR.l.signal, orig_sr=fs_h, target_sr=fs_min)
    HRIR.r.signal = resample(HRIR.r.signal, orig_sr=fs_h, target_sr=fs_min)
    print('resampled')
elif fs_s < fs_min:
    s = resample(s, orig_sr=fs_s, target_sr=fs_min)
    print('resampled')
else:
    print('non resampled')

# for plots
freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

'''Preprocessing of the HRIR, according to paper [11], allpass reducing ITD at high frequencies to code a higher 
order even with a lower order '''
# TODO(implement refinement)

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

# Plots of the filter magnitude and phase
if HRTF_refinement_plot:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    positionIndex = 10000
    ax1.plot(freq, 10 * np.log10(abs(allPassFilter_l[positionIndex, 0:NFFT // 2 + 1])))
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_ylim(-1, 1)
    ax1.set_title('Spectrum')

    ax2.plot(freq, np.unwrap(np.angle(allPassFilter_l[positionIndex, 0:NFFT // 2 + 1])), label="l")
    ax2.plot(freq, np.unwrap(np.angle(allPassFilter_r[positionIndex, 0:NFFT // 2 + 1])), label="r")
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_title('Phase')
    plt.legend()

# fft
HRTF_L = np.fft.rfft(HRIR.l.signal, NFFT)  # taking only the components [:, 0:int(NFFT / 2 + 1)] of the regular np.fft
HRTF_R = np.fft.rfft(HRIR.r.signal, NFFT)
if HRTF_refinement_plot:
    ax3.plot(freq, np.unwrap(np.angle(HRTF_L[positionIndex])) - np.unwrap(np.angle(HRTF_R[positionIndex])),
             label='Difference LR in phases - non filtered')
    # ax3.plot(freq, np.unwrap(np.angle(HRTF_R[positionIndex])), label='non filtered R')
# applying the filter
if HRTF_refinement:
    HRTF_L = HRTF_L * allPassFilter_l
    HRTF_R = HRTF_R * allPassFilter_r
if HRTF_refinement_plot:
    ax3.plot(freq, np.unwrap(np.angle(HRTF_L[positionIndex])) - np.unwrap(np.angle(HRTF_R[positionIndex])),
             label='Difference LR in phases - filtered')
    # ax3.plot(freq, np.unwrap(np.angle(HRTF_R[positionIndex])), label='filtered R')
    ax3.set_xscale('log')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase comparison HRTF')
    plt.legend()
    plt.show()
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
# DRFR = np.fft.fft(DRIR, NFFT)[:, 0:int(NFFT / 2 + 1)]  # taking only the first part of the FFT, not the reversed part

DRFR = np.fft.rfft(DRIR, NFFT)

Pnm = spatialFT(  # Pnm : Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
    DRFR,
    grid,
    grid_type='cart',
    order_max=N,
    kind="complex",
    spherical_harmonic_bases=None,
    weight=None,
    leastsq_fit=True,
    regularised_lstsq_fit=False  # not nice result, don't know why
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
'Old code'
# temp = np.transpose(weights(N, kr, 'rigid')) # cannot use the weight function if cannot integrate directly on the points (special grid)
# i = kr[kr == N]
# temp[temp == 0] = 1e-12
# max_i = np.min([i for i, n in enumerate(kr) if n > N])
#
# # weighting of the radial filter as described in ref [13]
# """This might be a little bit like doing a regularization in the method"""
# amp_max = 10 ** (amp_maxdB / 20)
# limiting_factor = (
#         2
#         * amp_max
#         / np.pi
#         * np.abs(temp)
#         * np.arctan(np.pi / (2 * amp_max * np.abs(temp)))
# )
# dn = limiting_factor / temp
# dn[:, max_i:] = 1 / temp[:, max_i:]  # do not limit the filter for higher frequencies
'End of old code'

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

# TODO(see if this works, because it probably reverses the phase, maybe need to relinearize the phase after)
# dn = utils.zero_pad_fd(dn, target_length_td=NFFT)


# plt.plot(np.unwrap(np.angle(dn[1])))
plt.plot(np.angle(dn[1]))
# # plotting the phases
# fig, (ax1, ax2) = plt.subplots(1, 2)  # plots
# for i in range(0, len(dn)):
#     ax1.plot( 20 * np.log10(abs(dn[i])), label='N = ' + str(i))
#     ax2.plot( np.unwrap(np.angle(dn[i])), label='N = ' + str(i))
# ax1.set_xscale('log')
# # ax.set_yscale('log')
# ax1.set_xlim(np.min(freq) + 1, np.max(freq))
# # ax.set_ylim(np.min(abs(dn[1])), np.max(abs(dn[1])))
# plt.legend()
# ax1.set_xlabel('Frequency (Hz)')
# ax2.set_xlabel('Frequency (Hz)')
# ax1.set_ylabel('Amplitude (dB)')
# ax2.set_ylabel('Phase (rad)')
# ax1.set_title('Spectrum before eq')
# ax2.set_title('Phase')
# plt.show()

# improvement EQing the orders that are truncated
if eq:
    dn_shf = gen.spherical_head_filter_spec(
        max_order=N,
        NFFT=NFFT,
        fs=fs_min,
        radius=r,
        is_tapering=tapering_win,
    )

    # make Spherical Head Filter causal -> apparently no need because already causal

    # dn_shf_delay_samples = NFFT / 2
    # dn_shf *= gen.delay_fd(
    #     target_length_fd=dn_shf.shape[-1], delay_samples=dn_shf_delay_samples
    # )

    dn[:] *= dn_shf

# improvement : tapering
if tapering_win:  # half sided hanning window
    w_tap = gen.tapering_window(N)
    dn = np.transpose(np.transpose(dn)*w_tap)

# plotting the radial filters, the phase has to be linear (causal filter)
if radial_plot:
    fig, (ax1, ax2) = plt.subplots(1, 2)  # plots
    for i in range(0, len(dn)):
        ax1.plot(freq, 20 * np.log10(abs(dn[i])), label='N = ' + str(i))
        ax2.plot(freq, np.unwrap(np.angle(dn[i])), label='N = ' + str(i))
    ax1.set_xscale('log')
    # ax.set_yscale('log')
    ax1.set_xlim(np.min(freq) + 1, np.max(freq))
    # ax.set_ylim(np.min(abs(dn[1])), np.max(abs(dn[1])))
    plt.legend()
    ax1.set_xlabel('Frequency (Hz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (dB)')
    ax2.set_ylabel('Phase (rad)')
    ax1.set_title('Spectrum')
    ax2.set_title('Phase')
    plt.show()

'''# SH domain convolution'''
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

    # chech that summation is 1
    # w2 = win[Nshift:] + win[:Nshift]
    # plt.plot(w2)

    # processing, mimic real time
    for n_sample in range(0, s_sampled_size[1] - 1):
        # sampling and windowing
        s_sampled[:, n_sample] = win * s[n_sample * Nshift:n_sample * Nshift + Nwin]
        # convolution TODO(checker mais probablement sr manquant)
        sl_out_sampled[:, n_sample], sr_out_sampled[:, n_sample] = np.transpose(
            scipy.signal.fftconvolve(s_sampled[:, n_sample], sl, mode='same')), np.transpose(
            scipy.signal.fftconvolve(s_sampled[:, n_sample], sr, mode='same'))
        # overlap add
        sl_out[n_sample * Nshift:n_sample * Nshift + Nwin] += win * sl_out_sampled[:,
                                                                    n_sample]  # apply the second window to have a smooth link between the samples
        sr_out[n_sample * Nshift:n_sample * Nshift + Nwin] += win * sr_out_sampled[:, n_sample]

    sl_out = sl_out[:-Nshift]  # to match the size of the input signal
    sr_out = sr_out[:-Nshift]
''' Scaling / Amplification '''
max = np.maximum(np.max(np.abs(sl_out)), np.max(np.abs(sr_out)))
sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''
sf.write('./exports/{0} rot={1} limiting={2}NFFT={3} realtime={4} HRTFmodif={5} Tapering={6} EQ={7}.wav'.format(
    output_file_name,
    str(rotation_sound_field_deg),
    str(amp_maxdB), str(NFFT),
    str(real_time),
    str(HRTF_refinement),
    str(tapering_win),
    str(eq)),
         np.stack((sl_out, sr_out), axis=1), fs_min)

"""
References
----------
[11] : Zaunschirm, M., Schörkhuber, C., & Höldrich, R. (2018). Binaural rendering of Ambisonic signals by head-related impulse response time alignment and a diffuseness constraint. The Journal of the Acoustical Society of America, 143(6), 3616-3627
[13] : Bernschütz, B. (2016). Microphone arrays and sound field decomposition for dynamic binaural recording. Technische Universitaet Berlin (Germany)
"""
