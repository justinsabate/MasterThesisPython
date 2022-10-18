############################
# Try to use SOFIA toolbox #
############################
# notebook accessible at this link : https://nbviewer.org/github/AppliedAcousticsChalmers/sound_field_analysis-py/blob/971cd1b62053afcdb4caa758d00af05affb48f86/examples/Exp4_BinauralRendering.ipynb
# this automatically reloads the sfa-py package if anything is changed there
import h5py
# Requirements
from IPython import display
import numpy as np

import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Load the local sound_field_analysis-py package
import sys

from code_SH.SphHarmUtils import spatialFT, get_eigenmike_grid, weights

sys.path.insert(0, "database/")
from sound_field_analysis import io, gen, process, plot, sph, utils



is_load_sofa    = True   # whether SOFA source data should be loaded, MIRO source data otherwise
sh_max_order    = 4      # maximal utilized spherical harmonic rendering order
is_real_sh      = False  # whether real spherical harmonic data type should be used (complex otherwise)
rf_nfft         = 4096   # target length of radial filter in samples
rf_amp_max_db   = 20     # soft limiting level of radial filter in dB
is_apply_rfi    = True  # not doing much   # whether radial filter improvement [1] should be applied
is_apply_sht    = True  # whether Spherical Harmonic Tapering [2] should be applied
is_apply_shf    = True  # whether Spherical Head Filter [3] should be applied
shf_nfft        = 256    # target length of Spherical Head Filter in samples (if applied)
azim_offset_deg = -140    # azimuth head rotation offset in degrees (to make the listener face the sound source)
pre_SSR_file    = "SSR_IRs.wav"  # target file for rendered preview SSR BRIRs
pre_azims_deg   = [-30]    # azimuth head orientations in degrees for rendered preview BRIR plots and auralizations
pre_len_s       = 20            # length of rendered preview BRIR auralizations in seconds
pre_src_file = "./wavfiles/BluesA_GitL.wav"  # audio source file for rendered preview BRIR auralizations

#################
### Load Data ###
#################
if is_load_sofa:
    # load impulse responses from SOFA file
    DRIR = io.read_SOFA_file("./database/RIR TH Koln/DRIR_CR1_VSA_110RS_R.sofa")
    HRIR = io.read_SOFA_file("./database/HRIR TH Koln/HRIR_L2702.sofa")
    FS = int(HRIR.l.fs)
    NFFT = HRIR.l.signal.shape[-1]
else:
    # added by Justin
    print('Miro files instead of sofa, more suited to use matlab')

# check match of sampling frequencies
if DRIR.signal.fs != FS:
    raise ValueError('Mismatch between sampling frequencies of DRIR and HRIR.')

# automatically calculate target processing length
# by summing impulse response lengths of DRIR, HRIR and radial filters
NFFT += DRIR.signal.signal.shape[-1] + rf_nfft
if is_apply_shf:
    NFFT += shf_nfft

##################
### Processing ###
##################
sh_kind = "real" if is_real_sh else "complex"

# HRIR SH coefficients
if is_load_sofa:
    # transform SOFA data
    # noinspection PyUnboundLocalVariable
    Hnm = np.stack(
        [
            process.spatFT(
                process.FFT(HRIR.l.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind=sh_kind,
            ),
            process.spatFT(
                process.FFT(HRIR.r.signal, fs=FS, NFFT=NFFT, calculate_freqs=False),
                position_grid=HRIR.grid,
                order_max=sh_max_order,
                kind=sh_kind,
            ),
        ]
    )
else:
    print('Miro files instead of sofa, more suited to use matlab')

# DRIR SH coefficients
print(NFFT)
Pnm = process.spatFT(
    process.FFT(DRIR.signal.signal, fs=FS, NFFT=NFFT, calculate_freqs=False), # Seems like no windowing done at all in here
    position_grid=DRIR.grid,
    order_max=sh_max_order,
    kind=sh_kind,
)

# # trying to compute the same as the other code, loading the measurements
# measurementFileName = 'database/DRIR anechoic Xenofon/AnechoicDataEigenmike.hdf5'
# channel = 0
# with h5py.File(measurementFileName, "r") as f:
#     measurementgroupkey = list(f.keys())[0]
#     DRIR = f[measurementgroupkey]['RIRs'][channel, :,
#            :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
#     MetaDatagroupkey = list(f.keys())[1]
#     fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency
#     # NFFT = np.shape(DRIR)[-1] # too big
#     f.close()
#
# grid = get_eigenmike_grid(plot=False)
# # Taking 100ms of the RIR signal
# nstart = 49000
# nend  = int(nstart+0.05*48000)
# DRIR = DRIR[:, nstart:nend]
#
# DRFR = np.fft.rfft(DRIR, NFFT)
#
# Pnm = spatialFT(  # Pnm : Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in columns
#     DRFR,
#     grid,
#     grid_type='cart',
#     order_max=4,
#     kind="complex",
#     spherical_harmonic_bases=None,
#     weight=None,
#     leastsq_fit=True,
#     regularised_lstsq_fit=False  # not nice result, don't know why
# )
# freq = np.arange(0, NFFT // 2 + 1) * (48000 / NFFT)
#
# r = 0.084 / 2
# c = 343
# kr = 2 * np.pi * freq * r / c
# nm_size = np.shape(Pnm)[0]
#
# temp = np.transpose(weights(4, kr, 'rigid'))
# i = kr[kr == 4]
# temp[temp == 0] = 1e-12
# max_i = np.min([i for i, n in enumerate(kr) if n > 4])
#
# # weighting of the radial filter as described in ref [13]
# """This might be a little bit like doing a regularization in the method"""
# amp_max = 10 ** (18 / 20)
# limiting_factor = (
#         2
#         * amp_max
#         / np.pi
#         * np.abs(temp)
#         * np.arctan(np.pi / (2 * amp_max * np.abs(temp)))
# )
# dn = limiting_factor / temp
# dn[:, max_i:] = 1 / temp[:, max_i:]  # do not limit the filter for higher frequencies


# compute radial filters
dn = gen.radial_filter_fullspec(
    max_order=sh_max_order,
    NFFT=rf_nfft,
    fs=FS,
    array_configuration=DRIR.configuration,
    amp_maxdB=rf_amp_max_db,
)

if is_apply_rfi:
    # improve radial filters (remove DC offset and make casual) [1]
    dn, _, dn_delay_samples = process.rfi(dn, kernelSize=rf_nfft)
else:
    # make radial filters causal
    dn_delay_samples = rf_nfft / 2
    dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)

# show frequency domain preview of radial filters
dn_legend = list(f"n = {n}" for n in range(sh_max_order + 1))
plot.plot2D(
    dn,
    fs=FS,
    viz_type="LogFFT",
    title="Radial filters",
    line_names=dn_legend,
)

# # show time domain preview of radial filter
# plot.plot2D(
#     np.fft.irfft(dn),
#     fs=FS,
#     viz_type="Time",
#     title="Radial filters",
#     line_names=dn_legend,
# )

# adjust length of radial filters
'''this is reversing the phase'''
dn = utils.zero_pad_fd(dn, target_length_td=NFFT)

# adjust radial filter data shape according to SH order as grades
dn = np.repeat(dn, np.arange(sh_max_order * 2 + 1, step=2) + 1, axis=0)

if is_apply_sht:
    # compute Spherical Harmonic Tapering window [2]
    dn_sht = gen.tapering_window(max_order=sh_max_order)

    # show preview of Spherical Harmonic Tapering window
    plot.plot2D(
        np.append(dn_sht, 1),  # append value which is not plotted
        viz_type="",
        title="Spherical Harmonic Tapering",
    )

    # adjust Spherical Harmonic Tapering data shape according to SH order as grades
    dn_sht = np.repeat(
        dn_sht[:, np.newaxis], np.arange(sh_max_order * 2 + 1, step=2) + 1, axis=0
    )

    # apply Spherical Harmonic Tapering to radial filters
    dn *= dn_sht

if is_apply_shf:
    # compute Spherical Head Filter [3]
    dn_shf = gen.spherical_head_filter_spec(
        max_order=sh_max_order,
        NFFT=shf_nfft,
        fs=FS,
        radius=0.085,#DRIR.configuration.array_radius
        is_tapering=is_apply_sht,
    )

    # make Spherical Head Filter causal
    dn_shf_delay_samples = shf_nfft / 2
    dn_shf *= gen.delay_fd(
        target_length_fd=dn_shf.shape[-1], delay_samples=dn_shf_delay_samples
    )

    # show frequency domain preview of Spherical Head Filter
    shf_title = "Spherical Head Filter"
    if is_apply_sht:
        shf_title += " (considering Spherical Harmonic Tapering)"
    # plot.plot2D(
    #     dn_shf,
    #     fs=FS,
    #     viz_type="LogFFT",
    #     title=shf_title,
    # )

    #     # show time domain preview of Spherical Head Filter
    #     plot.plot2D(
    #         np.fft.irfft(dn_shf),
    #         fs=FS,
    #         viz_type="Time",
    #         title=shf_title,
    #     )

    # adjust length of Spherical Head Filter
    dn_shf = utils.zero_pad_fd(dn_shf, target_length_td=NFFT)

    # apply Spherical Head Filter to radial filters
    dn *= dn_shf
    dn_delay_samples += dn_shf_delay_samples

# SH grades stacked by order
m = sph.mnArrays(sh_max_order)[0]

# reverse indices for stacked SH grades
m_rev_id = sph.reverseMnIds(sh_max_order)

# select azimuth head orientations to compute (according to SSR BRIR requirements)
azims_SSR_rad = np.deg2rad(np.arange(0, 360) - azim_offset_deg)

if is_real_sh:
    # compute possible components before the loop
    Pnm_dn = Pnm * dn

    # loop over all head orientations that are to be computed
    # this could be done with one inner product but the loop helps clarity
    S = np.zeros([len(azims_SSR_rad), Hnm.shape[0], Hnm.shape[-1]], dtype=Hnm.dtype)
    for azim_id, alpha in enumerate(azims_SSR_rad):
        alpha_cos = np.cos(m * alpha)[:, np.newaxis]
        alpha_sin = np.sin(m * alpha)[:, np.newaxis]

        # these are the spectra of the ear signals
        S[azim_id] = np.sum(
            (alpha_cos * Pnm_dn - alpha_sin * Pnm_dn[m_rev_id]) * Hnm, axis=1
        )
else:
    # compute possible components before the loop
    # apply term according to spherical harmonic kind
    Pnm_dn_Hnm = np.float_power(-1.0, m)[:, np.newaxis] * Pnm[m_rev_id] * dn * Hnm

    # loop over all head orientations that are to be computed
    # this could be done with one inner product but the loop helps clarity
    S = np.zeros([len(azims_SSR_rad), Hnm.shape[0], Hnm.shape[-1]], dtype=Hnm.dtype)
    for azim_id, alpha in enumerate(azims_SSR_rad):
        alpha_exp = np.exp(-1j * m * alpha)[:, np.newaxis]

        # these are the spectra of the ear signals
        S[azim_id] = np.sum(Pnm_dn_Hnm * alpha_exp, axis=1)

# IFFT to yield ear impulse responses
BRIR = process.iFFT(S)

# normalize BRIRs
BRIR *= 0.9 / np.max(np.abs(BRIR))

### Shoud not be necessary as we don't use SSR
# # generate grid compatible to SSR (1 degree steps on the horizon)
# BRIR_grid = io.SphericalGrid(
#     azimuth=azims_SSR_rad + np.deg2rad(azim_offset_deg),
#     colatitude=np.broadcast_to(np.deg2rad(90), azims_SSR_rad.shape),
# )
#
# # export file
# io.write_SSR_IRs(
#     filename=pre_SSR_file,
#     time_data_l=io.ArraySignal(
#         signal=io.TimeSignal(signal=BRIR[:, 0, :], fs=FS), grid=BRIR_grid
#     ),
#     time_data_r=io.ArraySignal(
#         signal=io.TimeSignal(signal=BRIR[:, 1, :], fs=FS), grid=BRIR_grid
#     ),
#     wavformat="float32",
# )
###

#####################
### Audio preview ###
#####################
# read source file
# source, source_fs = io.read_wavefile(pre_src_file)

# Needed to import it in float32, changed by justin
source, source_fs = librosa.load('./wavfiles/BluesA_GitL.wav', sr=None, mono=True,duration=20, dtype=np.float32)
if len(source.shape) > 1:
    source = source[0]  # consider only first channel

# select shorter extract from source
source = np.atleast_2d(source[: int(pre_len_s * source_fs)])

# resample source to match BRIRs
source = utils.simple_resample(source, original_fs=source_fs, target_fs=FS)

# show preview per specified orientation
for azim in pre_azims_deg:
    # changed by justin

    # display.display(
    #     display.Markdown(f"<h3>Head orientation {azim}°</h3>"),
    #     display.Audio(
    #         process.convolve(source, BRIR[azim]),
    #         rate=FS,
    #     ),
    # )
    sf.write('./exports/convolution'+str(azim)+'.wav', np.transpose(process.convolve(source, BRIR[azim])), FS)


####################
### Plot preview ###
####################
pre_legend = list(f"Left ear, {azim}°" for azim in pre_azims_deg)
pre_IR = list(BRIR[azim, 0] for azim in pre_azims_deg)
pre_TF = process.FFT(pre_IR, fs=source_fs, calculate_freqs=False)

# show time domain preview
# display.display(
#     display.Markdown(
#         f"<h3><center>Casual radial filter introduced delay of {dn_delay_samples:.0f} samples "
#         f"or {1e3 * dn_delay_samples / source_fs:.0f} ms.</center></h3>"
#     )
# )
# plot.plot2D(pre_IR, fs=source_fs, viz_type="Time", line_names=pre_legend)
# plot.plot2D(pre_IR, fs=source_fs, viz_type="ETC", line_names=pre_legend)

# show frequency domain preview
# plot.plot2D(pre_TF, fs=source_fs, viz_type="LogFFT", line_names=pre_legend)#, log_fft_frac=3)
###