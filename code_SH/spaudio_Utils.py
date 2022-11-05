# -*- coding: utf-8 -*-
"""Adapted function from library spaudio, the object-oriented code was changed to be better included
in the code signalProcessingFramework.

The library documentation can be found  at https://spaudiopy.readthedocs.io/en/latest/index.html"""

import numpy as np
from spaudiopy import sph

def magls_bin(hrirs, N_sph, f_trans=None, hf_cont='angle', hf_delay=(0, 0), fs=None, azi=None, elev=None,
              gridpoints=None, Nfft=4096, basis=None):
    """Magnitude Least-Squares (magLS) binaural decoder.

    This binaural decoder renders the (least squares) binaural output below
    `f_trans`, while rendering a magnitude solution above.


    Parameters
    ----------
    hrirs : sig.HRIRs
        HRIR set.
    N_sph : int
        Ambisonic (SH) order.
    f_trans : float, optional
        Transition frequency between linear and magLS handling.
        The default is None, which sets it to 'N_sph * 500'.
    hf_cont : ['delay', 'avg', 'angle'], optional
        High Frequency phase continuation method . The default is 'angle'.
    hf_delay : (2,), optional
        High frequency (additional) group delay in smpls. The default is (0, 0).


    Raises
    ------
    ValueError
        When passing not supported option.

    Returns
    -------
    hrirs_mls_nm : (2, (N_sph+1)**2, L)
        Decoding IRs matrix, 2: left,right (stacked), real coeffs, L taps.

    Notes
    -----
    The iterative procedure in [1] suffers form HF dispersion (available by
    `hf_cont='delay'` and `hf_delay=(0, 0))`.
    This function offers multiple options to mitigate this issue. E.g. manually
    estimating and setting `hf_delay`, or estimating a phase difference on
    previous frequency bins which are then used to predict.
    This delta can be the spherical average `hf_cont='avg'`, which offers an
    algorithmic way to estimate the global group delay.
    It can also be estimated per direction when `hf_cont='angle'`, which is
    able to preserve group delay changes over the angle, however, might
    reintroduce more complexity again.
    For very low orders, there might be a slight tradeoff.

    References
    ----------
    [1] Zotter, F., & Frank, M. (2019). Ambisonics. Springer Topics in Signal
    Processing.

    See Also
    --------
    :param Nfft:
    :py:func:`spaudiopy.decoder.sh2bin` : Decode Ambisonic streams to binaural.
    """
    # assert (isinstance(hrirs, sig.HRIRs))
    if f_trans is None:
        f_trans = N_sph * 500  # from N > kr
    '''Replaced by justin to match our measurements that are not using the same kind of objects'''
    # fs = hrirs.fs
    # hrirs_l = hrirs.left
    # hrirs_r = hrirs.right
    # azi = hrirs.grid['azi']
    # zen = hrirs.grid['colat']

    if len(np.shape(hrirs))==2: #TODO poor solution because going to do it twice for nothing
        hrirs_l = hrirs
        hrirs_r = hrirs
        print('Warning, mono signal, magls should use 2 channels to return Hnm coefficients')
    elif len(np.shape(hrirs))==3:
        hrirs_l = hrirs[0, :, :]
        hrirs_r = hrirs[1, :, :]
    azi = azi  # useless
    zen = elev
    gridpoints = gridpoints

    # numSmpls = hrirs.left.shape[1]
    numSmpls = hrirs_l.shape[1]

    nfftmin = Nfft
    # nfftmin = 1024
    nfft = np.max([nfftmin, numSmpls])
    freqs = np.fft.rfftfreq(nfft, 1 / fs)

    hrtfs_l = np.fft.rfft(hrirs_l, n=nfft)
    hrtfs_r = np.fft.rfft(hrirs_r, n=nfft)

    k_cuton = np.argmin(np.abs(freqs - f_trans))

    if basis is None:
        Y = sph.sh_matrix(N_sph, azi, zen, 'real')  # not sure about cartesian or spherical coordinates
    else:
        Y = basis  # is not real anymore,

    Y_pinv = np.linalg.pinv(Y)

    hrtfs_mls_nm = np.zeros((2, (N_sph + 1) ** 2, len(freqs)), dtype=hrtfs_l.dtype)
    '''Original code '''
    # phi_l_mod = np.zeros((hrirs.grid_points, len(freqs)))
    # phi_r_mod = np.zeros((hrirs.grid_points, len(freqs)))
    '''New code'''
    phi_l_mod = np.zeros((len(gridpoints[0]), len(freqs)))
    phi_r_mod = np.zeros((len(gridpoints[0]), len(freqs)))
    # TODO: weights, transition, order dependent (not my todo)

    # linear part
    hrtfs_mls_nm[0, :, :k_cuton] = Y_pinv @ hrtfs_l[:, :k_cuton]
    hrtfs_mls_nm[1, :, :k_cuton] = Y_pinv @ hrtfs_r[:, :k_cuton]
    phi_l_mod[:, :k_cuton] = np.angle(Y @ hrtfs_mls_nm[0, :, :k_cuton])
    phi_r_mod[:, :k_cuton] = np.angle(Y @ hrtfs_mls_nm[1, :, :k_cuton])

    if hf_cont == 'avg':
        # get the delta (from prediction frame)
        n_delta = 5
        assert (k_cuton > n_delta)
        # from spat avg
        delta_phi_l = np.mean(np.diff(np.unwrap(np.angle(
            hrtfs_mls_nm[0, 0, :k_cuton]))[k_cuton - n_delta - 1:k_cuton - 1]))
        delta_phi_r = np.mean(np.diff(np.unwrap(np.angle(
            hrtfs_mls_nm[1, 0, :k_cuton]))[k_cuton - n_delta - 1:k_cuton - 1]))
    elif hf_cont == 'angle':
        # get the delta (from prediction frame)
        n_delta = 5
        assert (k_cuton > n_delta)
        # for each angle
        delta_phi_l = np.mean(np.diff(np.unwrap(
            phi_l_mod)[:, k_cuton - n_delta - 1:k_cuton - 1]), axis=-1)
        delta_phi_r = np.mean(np.diff(np.unwrap(
            phi_r_mod)[:, k_cuton - n_delta - 1:k_cuton - 1]), axis=-1)
    elif hf_cont == 'delay':
        delta_phi_l = 0
        delta_phi_r = 0
    else:
        raise ValueError("hf_cont method not implemented")
    # apply (additional) group hf delay
    delta_w = 2 * np.pi * freqs[1]
    delta_phi_l = delta_phi_l - delta_w * (hf_delay[0] / fs)
    delta_phi_r = delta_phi_r - delta_w * (hf_delay[1] / fs)

    # manipulate above k_cuton
    for k in range(k_cuton, len(freqs)):
        phi_l_mod[:, k] = np.angle(Y @ hrtfs_mls_nm[0, :, k - 1])
        phi_r_mod[:, k] = np.angle(Y @ hrtfs_mls_nm[1, :, k - 1])

        # forward predict phase
        phi_l_mod[:, k] = phi_l_mod[:, k] + delta_phi_l
        phi_r_mod[:, k] = phi_r_mod[:, k] + delta_phi_r

        # transform
        hrtfs_mls_nm[0, :, k] = Y_pinv @ (np.abs(hrtfs_l[:, k]) *
                                          np.exp(1j * phi_l_mod[:, k]))
        hrtfs_mls_nm[1, :, k] = Y_pinv @ (np.abs(hrtfs_r[:, k]) *
                                          np.exp(1j * phi_r_mod[:, k]))
    '''Original code : transforming back to time domain to get order dependent impulse responses'''
    # hrirs_mls_nm = np.fft.irfft(hrtfs_mls_nm)
    # hrirs_mls_nm = hrirs_mls_nm[..., :numSmpls]
    '''Removed because tapering implemented somewhere else'''
    # taper last samples
    # taper_taps = 32
    # hrirs_mls_nm[:, :, -taper_taps:] *= np.hanning(2 * taper_taps)[taper_taps:]
    # # taper first ?
    # hrirs_mls_nm[:, :, :4] *= np.hanning(8)[:4]

    if len(np.shape(hrirs)) == 2:
        return hrtfs_mls_nm[0, :, :]
    else:
        return hrtfs_mls_nm


