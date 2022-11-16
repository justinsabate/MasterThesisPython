"""Python script aimed at preprocessing RIR to have different effects on the early reflections"""
import h5py
from librosa import load
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from code_Utils.FilterUtils import firfilter, zerophasefilter, minimumphasefilter, plot_response
from code_Utils.MixTimeUtils import get_direct_index, data_based
from code_Utils.SamplingUtils import resample_if_needed

'''Code controls'''
# Plots
plot1 = 0
plot2 = 0

# Signal to be convolved
signal_name = 'BluesA_GitL'  # without file extension, in wavfiles folder
extension = '.wav'
start_time = 0
end_time = 10

# Filtering of early reflections
method = 'gain'  # mini, zero, fir, gain are the different possibilities
gain = 2
cutoff = 2000
trans_width = 200
filter_type = 'highpass'

'''If h5py file, need to extract a position and a channel form this'''
measurementFileName = './database/Measurements-10-oct/DataEigenmikeDampedRoom10oct.hdf5'
position = 7
# channel = 9
# outputFileName for convolved signal in case it is needed
outputFileName = '' + method + '_' + filter_type

# Loading mixing time and indextdirect to avoid recalculating them, have to be calculated for the right position for the tdirect
loadDRIR_filename = 'DRIRs_processed_pos'+str(position)+'_cut2000_width200_'+filter_type+'.npz'
load_avgmixtime_indextdirect = 1

'''If plain wav file'''
# file1 = 'BluesA_GitL zeros in the beginning rot=0 pos=11 limiting=18NFFT=4096 realtime=0 HRTFmodif=1 Tapering=1 EQ=1'
# extension = '.wav'
#
# s1, fs1 = load('./exports/' + file1 + extension, sr=None, mono=True, dtype=np.float32)


# todo : test multiple channels
with h5py.File(measurementFileName, "r") as f:
    measurementgroupkey = list(f.keys())[0]
    DRIRs = f[measurementgroupkey]['RIRs'][position, :,  # todo: channel replaced by 0:2
            :]  # RIR, 106 measurements of the 32 channels eigenmike x number of samples
    MetaDatagroupkey = list(f.keys())[1]
    fs_r = f[MetaDatagroupkey].attrs['fs']  # Sampling frequency

    f.close()

'''Constants for plots'''
t = np.arange(0, len(DRIRs[0])) / fs_r

'Initialization'
DRIRs_processed = np.copy(DRIRs)
list_tmp50 = []
list_indexDirect = []

'First loop to get the average mixing time, as it is for one room and to ' \
    'remove variability across channels + see paper [55] (justins numerotation)'

if not load_avgmixtime_indextdirect:
    for channel in range(len(DRIRs)):
        'Calculating mixing time'
        DRIR = DRIRs[channel]
        tmp50, indexDirect = np.array(data_based(np.transpose(DRIR), fs_r), dtype=object)[
            [0, 5]]  # double brackets to extract 2 values out of the function outputs
        'Storing values for the second loop'
        list_tmp50.append(tmp50)
        list_indexDirect.append(indexDirect)
        print('Chanel nb '+str(channel) + ', mixing time : '+str(tmp50) + ' ms')

    avg_mixtime = np.mean(list_tmp50)
else:
    loaded = np.load('database/' + loadDRIR_filename)
    avg_mixtime = loaded['avg_mixtime']
    list_indexDirect = loaded['list_indexDirect']

    # avg_mixtime *= 3  # todo: remove it, for now trying to see what happens if we remove more than just the early reflections


print('Average mixing time : '+str(avg_mixtime)+' ms')

'Second loop for early reflection processing'
for channel in range(len(DRIRs)):

    '''Getting stored values'''
    DRIR = DRIRs[channel]
    tmp50 = avg_mixtime
    indexDirect = list_indexDirect[channel]

    tDirect = indexDirect / fs_r
    tmp50_plot = tmp50 * 0.001 + tDirect

    # i_center = 7700
    # i_start = i_center-Nwin//2
    # i_end = i_center+Nwin//2

    i_start = indexDirect
    i_end = int(tmp50_plot * fs_r)

    '''Create the window'''
    lenWin = i_end - i_start
    Nwin = 512
    fade = np.hanning(Nwin)
    win = np.concatenate((fade[:Nwin // 2], np.ones(lenWin - Nwin), fade[Nwin // 2:]))

    '''Extracting the reflection(s)'''
    refl_extractor = np.concatenate((np.zeros(i_start), win, np.zeros((len(DRIR) - i_end))))
    refl = DRIR * refl_extractor

    'Modification of the reflections'
    refl_plot = np.copy(refl)

    # # Filtering those reflections

    if method == 'zero':
        refl = zerophasefilter(refl, cutoff, fs_r, filter_type=filter_type, plot=plot1)
        print('Zero phase filtering used')
    elif method == 'mini':
        refl = minimumphasefilter(refl, fs_r, cutoff, trans_width, filter_type, numtaps=513, plot=plot1)
        print('Minimum phase filtering used')
    elif method == 'gain':

        refl *= gain
    else:
        refl = firfilter(refl, fs_r, cutoff, trans_width, filter_type, numtaps=513, plot=plot1)
        print('Simple fir filter used, phase variations expected')

    'Plot the modification of the reflections'
    if plot1:
        'Time representation'
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, refl_plot, label='Extracted reflections')
        ax1.grid()
        ax1.set_xlabel('Time(s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        delta_t = (tmp50_plot - tDirect) / 4
        ax1.set_xlim(tDirect - delta_t, tmp50_plot + delta_t)

        ax2.plot(t, refl, label='Modified reflections', color='#ff7f0e')
        ax2.grid()
        ax2.set_xlabel('Time(s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        fig.suptitle('Early reflections - Time representation')

        'Frequency representation'
        h1 = np.fft.rfft(refl_plot)
        h2 = np.fft.rfft(refl)
        f = np.linspace(0, 1, len(h1))
        fig = plt.subplots(2, 1)[0]  # because subplots does not return a figure but a tuple with the figure inside
        fig, ax1, ax2 = plot_response(fs_r, 2 * np.pi * f, h1, 'Extracted reflections', fig, subplot=0, unwrap=True)
        ax1.set_ylim(-100, 5)
        # To study the phase, set unwrap to False to see something and uncomment the next 2 lines
        # ax2.set_ylim(-5, 5)
        # ax2.set_xlim(cutoff - trans_width/3, cutoff + trans_width/3)
        fig, ax1, ax2 = plot_response(fs_r, 2 * np.pi * f, h2, 'Modified reflections', fig, subplot=1, unwrap=True)
        ax1.set_ylim(-100, 5)
        # To study the phase, set unwrap to False to see something and uncomment the next 2 lines
        # ax2.set_ylim(-5, 5)
        # ax2.set_xlim(cutoff - trans_width/3, cutoff + trans_width/3)
        fig.suptitle('Early reflections - Frequency representation')

    'Extracting the rest'
    rest = DRIR * (1 - refl_extractor)

    'Reconstruction'
    recon = rest + refl

    '''Plot the window on the signal waveform'''
    if plot2:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(t, DRIR, label='Original RIR')

        ax11 = ax1.twiny()
        ax11.bar(tDirect,
                 height=2 * np.max(abs(DRIR)),
                 width=0.002,
                 tick_label='Direct sound: ' + str(tDirect)[0:5] + 'sec',
                 bottom=-np.max(abs(DRIR)),
                 color='red')
        ax11.tick_params(axis='x', colors='red', rotation=10)

        ax12 = ax1.twiny()
        ax12.bar(tmp50_plot,
                 height=2 * np.max(abs(DRIR)),
                 width=0.002,
                 tick_label='tmp50: ' + str(tmp50_plot)[0:5] + 'sec',
                 bottom=-np.max(abs(DRIR)),
                 color='green'
                 )
        ax12.tick_params(axis='x', colors='green', rotation=10)

        ax2.plot(t, refl_plot, label='extracted reflections')
        ax2.plot(t, rest, label='remaining RIR')

        ax3.plot(t, refl_extractor, label='window')

        ax4.plot(t, recon, label='processed RIR\n reflections filtered')

        axs = fig.get_axes()
        for i, ax in enumerate(axs):
            if i < (len(axs) - 2):  # exclude the bars
                ax.set_xlabel('Time(s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
            ax.grid()
            ax.set_xlim((0.1, 0.3))
        plt.draw()

    'Storing in the global variable'
    DRIRs_processed[channel] = recon

    print('Channel nb ' + str(channel) + ' processed')

'Saving the file as a numpy array'
filename = 'database/DRIRs_processed_'+'pos'+str(position)+'_cut'+str(cutoff)+'_width'+str(trans_width)+'_'+filter_type+'.npz'
np.savez_compressed(filename,
                    DRIRs_processed=DRIRs_processed,
                    fs_r=fs_r,
                    avg_mixtime=avg_mixtime,
                    list_indexDirect=list_indexDirect)
print('saving processed DRIR in the file '+filename)

'To not close the figures'
plt.show()

'''Convolution with dry signal to try to hear differences on the method to filter the early reflections'''
### Loading the anechoic signal to be convolved
# s, fs_s = load('./wavfiles/' + signal_name + extension, sr=None, mono=True, offset=start_time,
#                duration=end_time - start_time, dtype=np.float32)

### Resampling, for now everything is 48000
# fs_min = min(fs_s, fs_r)
# resample_if_needed(fs_min, fs_r, refl, fs_s, s)

### Convolution
# s_out = np.transpose(fftconvolve(refl, s))

### Normalization
# s_out = s_out/max(np.abs(s_out))

'''To export the processed RIR'''
# sf.write('./exports/{0}.wav'.format(
#     outputFileName),
#     s_out,  # recon expected here
#     fs_r
# )
