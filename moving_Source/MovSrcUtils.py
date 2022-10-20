import numpy as np
import scipy
def moving_convolution(BRIR, signal, speed):
    length = len(signal)
    Nwin = length//speed
    Nshift = Nwin//2
    nb_window = length//Nwin  # will crop the end
    window = np.sqrt(np.bartlett(Nwin)) # the sum of first half and second half is constant, slightly lower than 1 -> no gain fluctuation
    i = 0
    j = 0
    convolved = np.zeros(Nwin*nb_window)
    while i+Nwin < (length-length % Nwin):
        signal_part = signal[i:(i+Nwin)]
        convolved_part = np.transpose(scipy.signal.fftconvolve(signal_part,BRIR[j], mode="same")) # to keep the same size as signal_part
        convolved_part *= window
        convolved[i:(i+Nwin)] += convolved_part
        i += Nshift
        j += 1  # nb of current window
        j = j % np.shape(BRIR)[0]  # reset j to angle 1 of BRIR
    return convolved