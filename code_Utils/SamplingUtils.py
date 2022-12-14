from librosa import resample

def resample_if_needed(fs_min, fs_r, DRIR, fs_s, s, fs_h=48000,  HRIR_l_signal=None, HRIR_r_signal=None):
    """Function that downsamples all the signals to the lowest sampling frequency fs_min
    fs_r corresponds to DRIR,
    fs_h corresponds to HRIR signals, if no signal, set to 48000 so no influence on the rest
    fs_s corresponds to s"""


    if fs_r > fs_min:
        DRIR = resample(DRIR, orig_sr=fs_r, target_sr=fs_min)
        print('RIR resampled')
        if fs_h > fs_min:
            if HRIR_l_signal is not None:
                HRIR_l_signal = resample(HRIR_l_signal, orig_sr=fs_h, target_sr=fs_min)
            if HRIR_r_signal is not None:
                HRIR_r_signal = resample(HRIR_r_signal, orig_sr=fs_h, target_sr=fs_min)
            print('HRIR resampled')
            if fs_s > fs_min:
                s = resample(s, orig_sr=fs_s, target_sr=fs_min)
                print('signal resampled')
        elif fs_s > fs_min:
            s = resample(s, orig_sr=fs_s, target_sr=fs_min)
            print('signal resampled')
    elif fs_h > fs_min:
        if HRIR_l_signal is not None:
            HRIR_l_signal = resample(HRIR_l_signal, orig_sr=fs_h, target_sr=fs_min)
        if HRIR_r_signal is not None:
            HRIR_r_signal = resample(HRIR_r_signal, orig_sr=fs_h, target_sr=fs_min)
        print(' HRIR resampled')
        if fs_s > fs_min:
            s = resample(s, orig_sr=fs_s, target_sr=fs_min)
            print('signal resampled')
    elif fs_s > fs_min:
        s = resample(s, orig_sr=fs_s, target_sr=fs_min)
        print('signal resampled')
    else:
        print('non resampled')
    return DRIR, s, HRIR_l_signal, HRIR_r_signal