import numpy as np
import scipy


def overlap_add(Nwin, s, sl, sr):
    """Overlap add method for convolution of sl and sr with s mimicking real time processing"""
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

    return sl_out, sr_out
