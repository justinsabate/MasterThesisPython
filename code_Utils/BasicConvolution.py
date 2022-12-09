"""File to calculate a simple convolution between a room response and a signal, used to obtain the anchor of the
mushra test, ie the monaural auralization """

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

# Signal to be convolved

# signal_name = 'BluesA_GitL'  # without file extension, in wavfiles folder
extension = '.wav'
sampling_frequency = 32000  # below, one can clearly hear the difference
NFFT = 4096*4

start_time = 0
end_time = 10

room = 'dry'  #'dry' or 'reverberant'

if room == 'dry':
    measurementFileName = '../database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
else:
    measurementFileName = '/Volumes/Transcend/DTU/Thesis/measurements_novdatacode/EigenmikeRecord/CleanedDataset/DataEigenmike_MeetingRoom_25nov_justin_cleaned.hdf5' #truncated file without ref

position = 9
channel = 0  # channel selected for the convolution (randomly)
# normalization factors to be able to compare the different measurements between themselves
max_reverberant = 0.015  # for position 6
max_dry = 0.054  # for position 10
for i, signal_name in enumerate(['Reinhardt_all', 'The Emperor (danish)', 'Frequency (english)', 'DontMeanAthin_all']):

    with File(measurementFileName, "r") as f:
        measurementgroupkey = list(f.keys())[0]

        DRIR = f[measurementgroupkey]['RIRs'][position, channel,
               :]  # RIR, 106 measurements of the 32 positions eigenmike x number of samples
        # REFs = f[measurementgroupkey]['REFs'][position]

        if room == 'reverberant':
            DRIR = DRIR/max_reverberant
            max_out = [8.352107666489198, 5.078845738356131, 6.152484973901738, 5.107540265517482]  ### For normalizations

        else:
            DRIR = DRIR/max_dry
            max_out = [1.6348721848432668, 2.2581357469111696, 2.3240527509425637, 1.9074752211472386]  ### For normalizations

        MetaDatagroupkey = list(f.keys())[1]
        fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency
        # NFFT = np.shape(DRIR)[-1] # too big
        f.close()

    s, fs_s = load('../wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
                   duration=end_time - start_time, dtype=np.float32)

    # s = s/np.max(np.abs(s))


    fs_min = min([fs_r, fs_s, sampling_frequency])

    DRIR, s, HRIR_l_signal, HRIR_r_signal = resample_if_needed(fs_min, fs_r, DRIR, fs_s, s)
    # max_reverberant = np.max(DRIR)  # for position 6
    # max_dry = 0.054  # for position 10
    if room == 'dry':
        nstart = int(6400 / fs_r * fs_min)
    # nend = int(nstart + 0.1 * fs_min)

    # for meeting room 2
    # for fs = 32000
    if room == 'reverberant':
        nstart = 0

    nend = nstart+NFFT
    # for fs = 48000
    # nstart = 3500
    # nend = 3500+NFFT

    DRIR = DRIR[nstart:nend]


    sl_out, sr_out = np.transpose(fftconvolve(DRIR, s)), np.transpose(fftconvolve(DRIR, s))




    # normalization already done in the code

    # max = 0.107  # sampling freq : 32 kHz and damped room and NNFT 4096
    sl_out, sr_out = sl_out / max_out[i], sr_out / max_out[i]

    write('../exports/convolved {0} position={1} channel={2} room ={3}.wav'.format(
        signal_name,
        str(position),
        str(channel),
        room),
        np.stack((sl_out, sr_out), axis=1), fs_min)
    print('Binauralized signal written')