"""This file is a Python implementation of the code found at
https://www.ak.tu-berlin.de/menue/publications/open_research_tools/mixing_time_prediction/ This code is a matlab,
and free access implementation of paper [55] that uses paper [56] as an estimation of the mixing time (separating the
early reflections from the reverberant field of a Room Impulse Response (RIR)), and adds a perceptual regression
allowing to have a better estimate of the perceived mixing time, where only the calculated mixing time was given before.

Refs:
[55] Lindau, A., Kosanke, L., & Weinzierl, S. (2012). Perceptual evaluation of model-and signal-based predictors of
the mixing time in binaural room impulse responses. Journal of the Audio Engineering Society, 60(11), 887-898.
[56] Abel, J. S., & Huang, P. (2006, October). A simple, robust measure of reverberation echo density. In Audio
Engineering Society Convention 121. Audio Engineering Society. """

import numpy as np
import math


def data_based(IR, fs):
    # processing parameters

    N = 1024  # window length(see Abel & Huang, 2006)
    onset_threshold_dB = -30  # peak criterion(onset_threshold * maximum)
    peak_secure_margin = 100  # used to protect peak from being affected when eliminating time of flight

    # cut IR (from onset position to stop_time)
    IR, index_direct, nb_channel = cut(IR, fs, onset_threshold_dB, peak_secure_margin)

    t_abel = np.zeros(nb_channel)
    echo_dens = np.zeros((np.shape(IR)[0], nb_channel))

    for n in range(0, nb_channel):
        t_abel[n], echo_dens[:, n] = np.array(abel(IR[:, n], N, fs, peak_secure_margin),dtype=object)

    # tmp from regression equations
    tmp50 = 0.8 * t_abel - 8
    tmp95 = 1.77 * t_abel - 38

    # clip negative values
    if sum(tmp50 < 0):
        idx = np.where(tmp50 < 0)
        for g in range(0, len(idx)):
            tmp50[idx[g]] = 1

    if sum(tmp95 < 0):
        idx = np.where(tmp95 < 0)
        for g in range(0, len(idx)):
            tmp50[idx[g]] = 1

    # average perceptual mixing time over all channels
    if nb_channel > 1:
        t_abel_interchannel_mean = np.mean(t_abel)
        tmp50_interchannel_mean = 0.8 * t_abel_interchannel_mean - 8
        tmp95_interchannel_mean = 1.77 * t_abel_interchannel_mean - 38
    else:
        tmp50_interchannel_mean = []
        tmp95_interchannel_mean = []

    return tmp50, tmp95, tmp50_interchannel_mean, tmp95_interchannel_mean, echo_dens, index_direct


def abel(IR, N, fs, peak_secure_margin):
    s = np.zeros(np.shape(IR)[0])
    anz_s = np.zeros(np.shape(IR)[0])
    echo_dens = np.zeros(np.shape(IR)[0])

    if np.shape(IR)[0] < N:
        print('IR shorter than analysis window len (1024 samples). Provide at least an IR of some 100 msec.')
        return None

    for n in range(np.shape(IR)[0]):
        # window at the beginning(increasing window len)
        if n <= N // 2:
            # standard deviation
            s[n] = np.std(IR[0:n + N // 2 - 1])

            # number of tips outside the standard deviation

            anz_s[n] = sum(abs(IR[0:n + N // 2 - 1]) > s[n])

            # echo density
            echo_dens[n] = anz_s[n] / N

        # window in the middle (constant window length)
        if N // 2 < n <= np.shape(IR)[0] - N // 2:
            s[n] = np.std(IR[n - N // 2 - 1:n + N // 2 - 1])
            anz_s[n] = sum(abs(IR[n - N // 2 - 1:n + N // 2 - 1]) > s[n])
            echo_dens[n] = anz_s[n] / N

        # window at the  (decreasing window length)
        if n > np.shape(IR)[0] - N // 2:
            s[n] = np.std(IR[n - N // 2 - 1:np.shape(IR)[0]])
            anz_s[n] = sum(abs(IR[n - N // 2 - 1:np.shape(IR)[0]]) > s[n])
            echo_dens[n] = anz_s[n] / N

    # normalize echo density
    echo_dens = echo_dens / math.erfc(1 / np.sqrt(2))

    # transition point(Abel & Huang(2006))
    # (echo density first time greater than 1)
    d = np.min(np.where(echo_dens > 1))
    t_abel = (d - peak_secure_margin) / fs * 1000

    if np.size(t_abel) == 0:
        print('Mixing time not found within given temporal limits. Try again with extended stopping crtiterion.')
        return None

    return t_abel, echo_dens


def cut(IR, fs, onset_threshold_dB, peak_secure_margin):

    IR, index_direct, nb_channel = get_direct_index(IR, onset_threshold_dB)
    if index_direct <= peak_secure_margin:
        peak_secure_margin = 0  # ignore peak_secure_margin

    '''Can implement easily a way to remove the end of the response'''
    stop_time = int(len(IR) - (index_direct - peak_secure_margin))  # IR length from peak to end in samples

    IR_cut = np.zeros((stop_time, nb_channel))

    for j in range(nb_channel):
        IR_cut[:, j] = IR[
                       index_direct - peak_secure_margin: index_direct - peak_secure_margin + stop_time,
                       j]

    return IR_cut, index_direct, nb_channel

def get_direct_index(IR, onset_threshold_dB= -30):
    '''IR is returned because it was reshaped to be easier to process'''

    if len(np.shape(IR)) == 1:
        nb_channel = 1
    else:
        nb_channel = np.shape(IR)[1]

    # set IR to the right size to be able to process multi-channels IR
    IR = np.reshape(IR, (np.shape(IR)[0], nb_channel))

    # calculate linear value of onset threshold from dB value
    onset_threshold = 10 ** (onset_threshold_dB / 20)

    go_on = 1
    k = -1

    index_direct = np.zeros(nb_channel)
    # for all IR - channels, find position where onset threshold value is reached
    for i in range(0, nb_channel):
        MAX = np.max(abs(IR[:]))
        # for full length of channel, find peak position
        while go_on:
            k = k + 1
            if abs(IR[k, i]) > MAX * onset_threshold:
                index_direct[i] = k
                go_on = 0
        go_on = 1
        k = 0

    return IR, int(np.min(index_direct)), nb_channel

