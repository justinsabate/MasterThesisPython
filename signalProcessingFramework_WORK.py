# import sys

from scipy.signal import fftconvolve
from librosa import load, resample
from sound_field_analysis import io, process, gen, utils  # to read HRTF set : sofa file
from sound_field_analysis.io import ArrayConfiguration

from code_SH.SphHarmUtils import *
from h5py import File
from soundfile import write

from code_Utils.MovSrcUtils import moving_convolution
from code_Utils.RealTimeUtils import overlap_add
from code_Utils.SamplingUtils import resample_if_needed
from code_plots.plots_functions import plot_HRTF_refinement, plot_radial, plot_freq_output, plot_grid

# import time
# tic = time.perf_counter()

'''Order of the spherical harmonics'''
N = 4 # maximum with the em32 eigenmike

'''Selection of the measurements'''

room = 'dry' #'dry' or 'reverberant'

if room == 'dry':
    measurementFileName = 'database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
else:
    # measurementFileName = '/Volumes/Transcend/DTU/Thesis/measurements_novdatacode/EigenmikeRecord/CleanedDataset/DataEigenmike_MeetingRoom_25nov_cleaned.hdf5' # full file with reference too
    measurementFileName = '/Volumes/Transcend/DTU/Thesis/measurements_novdatacode/EigenmikeRecord/CleanedDataset/DataEigenmike_MeetingRoom_25nov_justin_cleaned.hdf5' #truncated file without ref

# signal_name = 'DontMeanAthin_all'  # without file extension, in wavfiles folder
signal_name = 'Frequency (english)'
extension = '.wav'
start_time = 0
end_time = 10

# dry :
#   close = position 10; GOOD
#   middle = position 9; GOOD
#   far  = position 7; BAD (low frequency boost, close to the closet in the room)
# reverberant :
#   close = position 6; GOOD
#   middle = position 3; (GOOD but in between)
#   far = position 0; GOOD

position = 10  # position of the measurement that is being used, (position 7 => -23° azimuth for dry environment)
# mixing time increase
increase_factor_window = 1

'''Rotation and head positioning : rotation around the listener if multiple values in rotation_sound_field_deg'''
offset = 0  # -23  # to position one specific measurement in front
# rotation_sound_field_deg = np.arange(-360, 361, 10) #np.arange(-90, 91, 5)
rotation_sound_field_deg = np.zeros(1)
rotation_sound_field_deg += offset  # to get it in front for position 7 with offset -23

'''Careful, loadHnm has to be set to 0 in case of changing the methods to obtain the Hnm coefficients'''
loadHnm = 1  # to load or calculate the Hnm coefficients, getting faster results

'''Loading preprocessed (==modified) DRIR or taking the measured one instead'''
processedDRIR = 0  # to load preprocessed DRIR obtained with the code ProcessRIR, if 0, not processed DRIR
filtertype = 'gain'  # gain, lowpass, highpass of threshold depending on the files generated in ProcessRIR
cutoff = 2000
if filtertype == 'gain':
    processedDRIR_filename = 'DRIRs_processed_' + room + '_pos' + str(position) + '_gain_mix*' + str(increase_factor_window) + '.npz'
elif filtertype == 'threshold':
    processedDRIR_filename = 'DRIRs_processed_' + room + '_pos' + str(position) + '_threshold.npz'
else:
    processedDRIR_filename = 'DRIRs_processed_'+room+'_pos'+str(position)+'_cut'+str(cutoff)+'_'+filtertype+'_zero.npz'


'''Exporting BRIR for metrics calculation'''
exportBRIR = 0

'''Convolving with the dry signal to write the wav binauralization'''
audiofileConvolve = 1

'''Algorithm choices, different refinements'''
real_time = 0
HRTF_refinement = 0  # from [11]
tapering_win = 1  # from sound field analysis toolbox + cf [36]
eq = 1  # from sound field analysis toolbox + cf [23], could probably be calculated from HRTF but calculated from a sphere (scattering calculations)
MLS = 1
is_apply_rfi = 0  # todo : maybe not put it back, weird results with some measurements but nice with others, depending on the content  # useful if MLS because a bit of unwanted stuff happening at low frequencies, but that is light

'''Plots choices'''
radial_plot = 0
freq_output_plot = 0
HRTF_refinement_plot = 0
grid_plot = 0
DRIR_plot = 0

'''Some initializations'''
amp_maxdB = 18  # for the radial filter see ref [13] ch 3.6.6 and 3.7.3.3
Nwin = 512  # for the real time processing, but no time difference, might use it in the future
sampling_frequency = 32000  # below, one can clearly hear the difference
output_file_name = signal_name

'''Mandatory conditions'''
if MLS:
    HRTF_refinement = 0

if HRTF_refinement == 0:
    HRTF_refinement_plot = 0

if np.size(rotation_sound_field_deg) > 1:
    if real_time == 1:
        real_time = 0
        print('Real time set to 0, case with real time and many directions not implemented')

'''Get the data'''
# Be sure they are the same type (float32 for ex)

### Measurement data
#
# positions = [0,1,2,3]
# plt.figure()

# for position in positions:
if not processedDRIR:
    print('Loading measured (non modified) DRIR')
    with File(measurementFileName, "r") as f:
        measurementgroupkey = list(f.keys())[0]

        DRIR = f[measurementgroupkey]['RIRs'][position, :,
               :]  # RIR, 106 measurements of the 32 positions eigenmike x number of samples
        # REFs = f[measurementgroupkey]['REFs'][position]

        MetaDatagroupkey = list(f.keys())[1]
        fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency
        # NFFT = np.shape(DRIR)[-1] # too big
        f.close()
else:
    print('Loading modified DRIR')
    loaded = np.load('database/'+processedDRIR_filename)
    DRIR = loaded['DRIRs_processed']
    fs_r = loaded['fs_r']
    avg_mixtime = loaded['avg_mixtime']
    #
    # # ### work in progress some plots todo:remove it
    # #
    #     # for i in range(0, np.shape(DRIR)[0]):
#     #     #     plt.plot(DRIR[i])
# plt.plot(REFs[0])
# plt.plot(DRIR[0])
#
# #
# plt.draw()


# maxiref = np.argmax(REFs)
# maxirir = np.argmax(DRIR[2])

### end of work in progress

grid = get_eigenmike_grid(plot=False)
plot_grid(grid, grid_plot)

### HRTF set extraction
if not loadHnm:
    HRIR = io.read_SOFA_file(
        "./database/HRIR TH Koln/HRIR_L2702.sofa")  # changed for MLS"./database/HRIR TH Koln/HRIR_FULL2DEG.sofa"
    fs_h = int(HRIR.l.fs)
    HRIR_l_signal = HRIR.l.signal
    HRIR_r_signal = HRIR.r.signal
else:
    print('loading saved Hnm coefficients')
    loaded = np.load('database/Hnm.npz')
    fs_h = loaded['fs_h']
    HRIR_l_signal = None
    HRIR_r_signal = None

### Loading the anechoic signal to be convolved
s, fs_s = load('./wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
               duration=end_time - start_time, dtype=np.float32)

'''Data preparation'''
### set to mono
if len(np.shape(s)) > 1:
    s = (s[0] + s[1]) / 2  # needs to be mono-
    # s = s[1]

fs_min = min([fs_r, fs_h, fs_s, sampling_frequency])
print('Sampling frequency :'+str(fs_min)+' Hz')
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
# HRIR_l_signal can be None
DRIR, s, HRIR_l_signal, HRIR_r_signal = resample_if_needed(fs_min, fs_r, DRIR, fs_s, s, fs_h, HRIR_l_signal,
                                                           HRIR_r_signal)

### Choosing NFFT, important parameter because it truncates the frequency responses so the room response
NFFT = 4096*4 # todo: changed from 4096
print('Nfft=' + str(NFFT))

### Taking NFFT/fs_min sec of the RIR signal
# for damped room with NFFT = 4096 and fs = 32000
# first index to be sure to have the response, depends on the measurements mostly, ideally would not have to change nstart = 0
if room == 'dry':
    nstart = int(6400 / fs_h * fs_min)
# nend = int(nstart + 0.1 * fs_min)

# for meeting room 2
# for fs = 32000
if room == 'reverberant':
    nstart = 0

nend = nstart+NFFT
# for fs = 48000
# nstart = 3500
# nend = 3500+NFFT

DRIR = DRIR[:, nstart:nend]

### studying the effect of the truncation in the beginning
if DRIR_plot:
    plt.plot(DRIR[9])
    plt.show()



### abcissa for plots
freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

'''Preprocessing of the HRIR, according to paper [11], allpass reducing ITD at high frequencies to code a higher 
order even with a lower order '''

### creating the filter
fc = 1500  # empirically chosen in [11], consistent with duplex theory
c = 343
if not loadHnm:
    if HRTF_refinement:
        rh = 8.5e-2  # HRIR.grid.radius
        azi = HRIR.grid.azimuth  # already spherical
        elev = HRIR.grid.colatitude
        tau_r = -np.cos(azi) * np.sin(elev) * rh / c
        tau_l = -tau_r

        # NFFT = 2048
        # freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)

        # Not actual compensation by the filter but instead an estimated one according to paper [11] (estimation on a
        # sphere, instead of the actual head compensation), it is said to have no difference perceptually and it is
        # more simple
        allPassFilter_r = np.ones((len(tau_r), NFFT // 2 + 1),
                                  dtype=np.complex128)  # shape : nb channels X frequency bins
        allPassFilter_l = np.ones((len(tau_l), NFFT // 2 + 1), dtype=np.complex128)

        for i, f in enumerate(freq):
            if f > fc:
                allPassFilter_r[:, i] = np.exp(
                    -1j * 2 * np.pi * (f - fc) * tau_r)
                allPassFilter_l[:, i] = np.exp(-1j * 2 * np.pi * (f - fc) * tau_l)

    ### fft
    HRTF_L = np.fft.rfft(HRIR_l_signal,
                         NFFT)  # taking only the components [:, 0:int(NFFT / 2 + 1)] of the regular np.fft
    HRTF_R = np.fft.rfft(HRIR_r_signal, NFFT)

    ### applying the filter
    if HRTF_refinement:
        HRTF_L = HRTF_L * allPassFilter_l
        HRTF_R = HRTF_R * allPassFilter_r

        # Plots of the filter magnitude and phase
        plot_HRTF_refinement(freq, allPassFilter_l, allPassFilter_r, HRTF_L, HRTF_R, NFFT, HRTF_refinement_plot)

''' SH expansion '''

if not MLS:
    if not loadHnm:
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
        np.savez_compressed('database/Hnm.npz', Hnm=Hnm, fs_h=fs_h)
        print('saving Hnm coefficients')
    else:
        Hnm = loaded['Hnm']
else:
    if not loadHnm:
        Hnm = spatialFT(
            np.stack((HRIR_l_signal, HRIR_r_signal), 0),
            HRIR.grid,
            grid_type='sphe',
            order_max=N,
            kind="complex",
            spherical_harmonic_bases=None,
            weight=None,
            leastsq_fit=False,
            regularised_lstsq_fit=False,
            MLS=True,
            fs=fs_min,
            NFFT=NFFT
        )
        np.savez_compressed('database/Hnm.npz', Hnm=Hnm, fs_h=fs_h)
        print('saving Hnm coefficients')
    else:
        Hnm = loaded['Hnm']

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
    regularised_lstsq_fit=False,
    MLS=False,
)

''' Radial filter + smoothing of the coefficients'''

# NFFT has an influence on the filter shape (low frequencies)

r = 0.084 / 2
c = 343
kr = 2 * np.pi * freq * r / c
nm_size = np.shape(Pnm)[0]

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
'''Taken from sound field analysis (sofia) toolbox, could not find the papers'''
if is_apply_rfi:
    # improve radial filters (remove DC offset and make casual) [1]
    dn, _, dn_delay_samples = process.rfi(dn, kernelSize=NFFT, highPass=0.0065)  # zero phase high pass
    # already making the filter causal in the rfi function
else:
    # make radial filters causal
    dn_delay_samples = NFFT / 2
    dn *= gen.delay_fd(target_length_fd=dn.shape[-1], delay_samples=dn_delay_samples)


''' Implementing the different improvements of the auralized signal '''
### improvement EQing the orders that are truncated cf paper [23]
# takes into account tapering or not (adapts to it)
if eq:
    dn_shf = gen.spherical_head_filter_spec(
        max_order=N,
        NFFT=NFFT,
        fs=fs_min,
        radius=r,
        is_tapering=False,  # otherwise it is done twice
    )

    dn_shf_delay_samples = NFFT / 2
    dn_shf *= gen.delay_fd(
        target_length_fd=dn_shf.shape[-1], delay_samples=dn_shf_delay_samples
    )

    dn[:] *= dn_shf  # effect : reversing the phase, and we can indeed hear a big difference, the phase is linear but
    # creates the delay necessary not to have weird "reflections" (with a hearable delay)
else:
    dn_shf_delay_samples = NFFT / 2
    dn *= gen.delay_fd(
        target_length_fd=dn.shape[-1], delay_samples=dn_shf_delay_samples
    )

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

### Complex convolution : Has to take the opposite of the m but not the n cf paper [12] (justin's references)
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

' Head tracking export trials '
# To export the channels corresponding to all directions

# import scipy
# scipy.io.savemat('exports/BRIR/BRIR_l.mat', dict(sl=np.transpose(sl)))
# scipy.io.savemat('exports/BRIR/BRIR_r.mat', dict(sr=np.transpose(sr)))

# Try to resample now to see if difference or not, need for the shortest BRIR for head tracking TODO(find new solution, cannot do this)
# sl = resample(sl, orig_sr=fs_min, target_sr=16000)
# sr = resample(sr, orig_sr=fs_min, target_sr=16000)
#
# s = resample(s,fs_min,16000)
# print('Resampling in the end done')
# fs_min = 16000  # for export
#

'Export of BRIR for metric calculation'
if exportBRIR:
    BRIR_filename = 'BRIR'
    write('./exports/BRIR/{0} pos={1} preprocessed={2} room={3} mix*{4}.wav'.format(
        BRIR_filename,
        str(position),
        str(processedDRIR),
        room,
        str(increase_factor_window)),
        np.stack((sl[0], sr[0]), axis=1), fs_min)
    print('BRIR file written')

''' Convolution with dry signal '''
if audiofileConvolve :
    speed = len(rotation_sound_field_deg) // 2  # this parameter encodes the speed of the change form one direction to
    speed *= 2

    if speed:  # moving source
        sl_out = moving_convolution(sl, s, speed)  # sl is of shape [nb of directions x signal size]
        sr_out = moving_convolution(sr, s, speed)
    else:  # static source
        # remove 2 dimensionality
        sl = sl[0]
        sr = sr[0]
        if not real_time:
            sl_out, sr_out = np.transpose(fftconvolve(sl, s)), np.transpose(fftconvolve(sr, s))
        else:  # taking samples of the signal and overlapping the result (no effect for now)
            sl_out, sr_out = overlap_add(Nwin, s, sl, sr)

''' Scaling / Amplification '''
'The max value was calculated according to the highest amplitude in the output of the algorithm, for position 11 of ' \
'measurements DataEigenmikeDampedRoom10oct.hdf5, ' \
'it depends on the resampling, usually if we lower the sampling frequency it has to go up'

# ## Regular use of it, measured on the position that leads to the biggest output and sets it (it is also
# sampling_frequency dependent
if audiofileConvolve :
    if fs_min == 32000:
        max = 0.107  # sampling freq : 32 kHz and damped room and NNFT 4096
        # max = 0.01# sampling freq : 32kHz and Meeting room 2 and NFFT 2*4096
        # max = 0.05
        # max = 0.0001  # sampling freq : 48kHz and Meeting room 2 and NFFT 4*4096
    elif fs_min == 48000:
        max = 0.209  # sampling freq : 48 kHz
    else:
        max = 0.209
    # needs to be constant to be able to encode the distance (have different gains in different positions)

    sl_out, sr_out = sl_out / max, sr_out / max

''' Writing file '''
if audiofileConvolve :
    write('./exports/{0} pos={1} preprocessed={2} room={3} {4} {5}.wav'.format(
        output_file_name,
        str(position),
        str(processedDRIR),
        room,
        filtertype*processedDRIR,
        str(cutoff)*processedDRIR),
        np.stack((sl_out, sr_out), axis=1), fs_min)
    print('Binauralized signal written')

# tac = time.perf_counter()
# print(tac-tic)


"""
References
----------
[11] : Zaunschirm, M., Schörkhuber, C., & Höldrich, R. (2018). Binaural rendering of Ambisonic signals by head-related impulse response time alignment and a diffuseness constraint. The Journal of the Acoustical Society of America, 143(6), 3616-3627
[13] : Bernschütz, B. (2016). Microphone arrays and sound field decomposition for dynamic binaural recording. Technische Universitaet Berlin (Germany)
"""
