from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def plot_response(fs, w, h, title):
    "Utility function to plot response functions"
    # # plot on separate figures
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)))
    # ax1.set_ylim(-60, 5)
    # ax1.set_xlim(0, 0.5 * fs)
    # ax1.grid(True)
    # ax1.set_xlabel('Frequency (Hz)')
    # ax1.set_ylabel('Gain (dB)')
    # ax1.set_title(title)
    #
    # ax2 = fig.add_subplot(212)
    # ax2.plot(0.5 * fs * w / np.pi, np.unwrap(np.angle(h)))
    # ax2.set_xlim(0, 0.5 * fs)
    # ax2.grid(True)
    # ax2.set_xlabel('Frequency (Hz)')
    # ax2.set_ylabel('Phase (rad)')

    # # plot on same figure
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)), color='#1f77b4')
    ax1.set_xscale('log')
    ax1.set_ylim(-40, 5)
    ax1.set_xlim(1, 0.5 * fs)
    ax1.grid(True)
    ax1.set_xlabel('Frequency (Hz)')

    ax1.set_ylabel('Gain (dB)', color='#1f77b4') #, fontsize='12'
    ax1.set_title(title)
    ax1.tick_params(axis='y', colors='#1f77b4') #, labelsize='12'

    ax2 = ax1.twinx()
    ax2.plot(0.5 * fs * w / np.pi, np.unwrap(np.angle(h)), color='#ff7f0e')
    # ax2.grid(True)
    # ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (rad)', color='#ff7f0e')
    ax2.tick_params(axis='y', colors='#ff7f0e') #, labelsize='12'





def firfilter(x, fs, cutoff, trans_width, filter_type, numtaps=400, plot=False, output_b=False):
    # Example for defining the parameters
    # fs = 22050.0  # Sample rate, Hz
    # cutoff = 8000.0  # Desired cutoff frequency, Hz
    # trans_width = 400  # Width of transition from pass band to stop band, Hz
    # numtaps = 400  # Size of the FIR filter.

    if filter_type == 'lowpass':
        b = signal.firls(numtaps, [0, cutoff, cutoff + trans_width, 0.5 * fs], [1, 1, 0, 0], fs=fs)
    elif filter_type == 'highpass':
        b = signal.firls(numtaps, [0, cutoff - trans_width, cutoff, 0.5 * fs], [0, 0, 1, 1], fs=fs)
    elif filter_type == 'bandpass':
        b = signal.firls(numtaps, [0, cutoff[0] - trans_width / 2, cutoff[0], cutoff[1], cutoff[1] + trans_width / 2,
                                      0.5 * fs], [0, 0, 1, 1, 0, 0], fs=fs)
    else:
        print('Filter type not known')
    if plot:
        w, h = signal.freqz(b, [1], worN=2000)
        plot_response(fs, w, h, filter_type + " filter")

    if output_b:
        return b
    else:
        # Use lfilter to filter x with the FIR filter.
        filtered_x = signal.lfilter(b, 1.0, x)
        return filtered_x

def zerophasefilter(x, cutoff, fs, filter_type='lowpass', plot=False):
    b, a = signal.butter(4, cutoff*2/fs, btype=filter_type)
    filtered_x = signal.filtfilt(b, a, x)
    if plot:
        w, h = signal.freqz(b, a, worN=2000)
        plot_response(fs, w, h, filter_type + " filter, applied twice to get the zero phase filter")
    return filtered_x

def minimumphasefilter(x, fs, cutoff, trans_width, filter_type, numtaps=400, plot=False):
    b = firfilter(x, fs, cutoff, trans_width, filter_type, numtaps, plot=False, output_b=True)
    b_minimum = signal.minimum_phase(b)
    filtered_x = signal.lfilter(b_minimum, 1.0, x)
    if plot:
        w, h = signal.freqz(b_minimum, [1], worN=2000)
        plot_response(fs, w, h, filter_type + " filter")
    return filtered_x