import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt


def CLL(s, NFFT):
    """Computes the Composite Loudness Level according to paper [39], itself citing [65]"""
    '''s is the stereo file, with 2 channels'''
    S = np.fft.rfft(s, NFFT)
    out = 10 * np.log10(np.abs(S[0, :]) ** 2 + np.abs(S[1, :]) ** 2)  # left and right channels, for all frequencies
    return out


def CLLerror(s, s_ref, NFFT=None):
    """Computes the CLL error according to paper [39], showing the differences in loudness and coloration between the
    binauralized signals s and the reference sref """
    if NFFT is None:
        NFFT = len(s)
    return CLL(s, NFFT) - CLL(s_ref, NFFT)


# def normInterCrossCorr(s, t1, t2, tau, fs):
#     """Returns the normalized interaural cross-correlation as defined in paper [66]"""
#     n1 = t1*fs
#     n2 = t2*fs
#     ntau = tau*fs
#     return np.sum(s[0, n1:n2]*s[1, n1+tau:n2+ntau])/np.sqrt(np.sum(s[0, n1:n2]**2)*np.sum(s[1, n1:n2]**2))


def IC(s, fs, NFFT):
    """Returns the frequency dependent interaural coherence as defined in paper [66], equation 2.4"""

    freqs, time, Zxx = scipy.signal.stft(s, fs=fs, nfft=NFFT)  # Zxx dimension channels X freqbins X timebins

    Zxxl = Zxx[0]  # left
    Zxxr = Zxx[1]  # right

    ic = np.real(np.sum(Zxxl * np.conj(Zxxr), 1)) / np.sqrt(np.sum(np.abs(Zxxl) ** 2, 1) * np.sum(np.abs(Zxxr) ** 2, 1))
    return ic


def ITD(BRIR, fs, plot=False):
    """Stated to be the best method for ITD estimation in paper [68] itself rating an implementationf of paper [69]"""

    threshold_dB = -30  # in dB
    threshold = 10 ** (threshold_dB / 20)
    cutoff = 3000
    filter_type = 'lowpass'

    b, a = signal.butter(6, cutoff * 2 / fs, btype=filter_type)  # because critical frequencies must be between 0 and
    # 1 1 being the nyquist frequency
    filtered_BRIR = signal.lfilter(b, a, BRIR)
    max_index = []
    t = np.linspace(0, (len(BRIR[0]) - 1) / fs, len(BRIR[0]))
    if plot:
        plt.figure()
    for channel in range(0, 2):
        # normalize both channels
        filtered_BRIR[channel, :] = filtered_BRIR[channel, :] / np.max(np.abs(filtered_BRIR[channel, :]))
        # get the first index with higher value than the threshold : it is out indicator for the first peak
        max_index.append(np.min([i for i, val in enumerate(np.abs(filtered_BRIR[channel, :])) if val > threshold]))
        if plot:
            plt.plot(t, filtered_BRIR[channel, :], label='normalized channel ' + str(channel))

    if plot:
        plt.plot(t, [threshold for i in filtered_BRIR[0, :]], label='threshold')  # straight line of the threshold
        plt.legend()
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title('BRIR normalized channels and threshold to calculate ITD')
        plt.draw()

    itd = (max_index[1] - max_index[0]) / fs

    return itd


def ILD(BRIR, fs, method='ERB'):
    f_low = 50
    f_high = 20000  # not always achievable because of fs
    NFFT = len(BRIR[0])
    BRTF = np.fft.rfft(BRIR, NFFT)

    '''cf torben poulsen acoustic communication for now, to be refined if needed'''

    if method == 'gammatone':
        nb = 39  # taken from paper [67]

    elif method == 'ERB':
        """Different comments : 
            - difference in number of bands, 39 bands stated in [67] VS Nb of bands calculated in the ERB scale
            - for now the formulas are taken from the course acoustic communication, and the same formulas are used in 
                wikipedia, a more accurate look for the papers is needed if htus method is used"""

        E = int(21.4 * np.log10(4.37e-3 * f_high + 1))  # number of bands needed
        nb = E

    fc = np.logspace(start=0, stop=np.log10(f_high), num=nb)
    # Trick to start at f = 50 HZ but still have the right number of ERB bands E
    for i, f in enumerate(fc):
        if f < f_low:
            start = i
        if f < fs / 2:  # to get the number of bands right (39 bands as in [67]) but not calculate bands higher than fs/2
            end = i
    fc = fc[start:end]
    ild = np.zeros(len(fc))

    if method == 'gammatone':
        # filters = np.zeros((len(fc), NFFT))

        for i, f in enumerate(fc):
            # Calculating the gammatone filter for a frequency f
            gammatone_b, gammatone_a = signal.gammatone(f, 'fir', numtaps=NFFT, fs=fs)
            # Filtering the 2 channels of the BRIR
            filtered_BRIR = signal.lfilter(gammatone_b, gammatone_a, BRIR)
            # Calculating the ild value for the frequency band
            filtered_BRTF = np.fft.rfft(filtered_BRIR)
            ild[i] = 10 * np.log10(np.sum(np.abs(filtered_BRTF[0]) ** 2) / np.sum(np.abs(filtered_BRTF[1]) ** 2))

    elif method == 'ERB':
        # Calculating the bandwidth of each frequency band centered around fc
        ERBs = 24.7 * (4.37e-3 * fc + 1)
        for i, f in enumerate(fc):
            bw = ERBs[i]
            index_low = int((f - bw / 2) * NFFT / fs)
            index_high = int((f + bw / 2) * NFFT / fs)
            # equivalent to filtering with rectangular window
            ild[i] = 10 * np.log10(
                np.sum(np.abs(BRTF[0, index_low:index_high]) ** 2) / np.sum(np.abs(BRTF[1, index_low:index_high]) ** 2))

    return fc, ild


"""
References
----------
[39] : Schörkhuber, C., Zaunschirm, M., & Höldrich, R. (2018, March). Binaural rendering of ambisonic signals via magnitude least squares. In Proceedings of the DAGA (Vol. 44, pp. 339-342).
[65] : Karjalainen, M., Ono, K., & Pulkki, V. (2001, November). Binaural Modeling of Multiple Sound Source Perception: Methodology and Coloration Experiments. In Audio Engineering Society Convention 111. Audio Engineering Society.
[24] : Rafaely, B., & Avni, A. (2010). Interaural cross correlation in a sound field represented by spherical harmonics. The Journal of the Acoustical Society of America, 127(2), 823-828.
[66] : Menzer, F. (2010). Binaural audio signal processing using interaural coherence matching (No. THESIS). EPFL.
[68] : Andreopoulou, A., & Katz, B. F. (2017). Identification of perceptually relevant methods of inter-aural time difference estimation. The Journal of the Acoustical Society of America, 142(2), 588-598
[69] : Algazi, R., Avendano, C., & Duda, R. O. (2001). Estimation of a spherical-head model from anthropometry. J. Aud. Eng. Soc., 49
"""
