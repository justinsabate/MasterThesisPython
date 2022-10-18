import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import linear_model
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift


def calc_scm(X):
    # X (n_freq,n_channels,n_timesteps) OR (n_freq,n_channels,n_snapshots,n_timesteps)
    if X.ndim == 4:
        return np.einsum('fnst,fmst->fnmt', X, np.conj(X), optimize=True) / X.shape[2]
    elif X.ndim == 3:
        return np.einsum('fnt,fmt->fnmt', X, np.conj(X), optimize=True)
    else:
        return np.einsum('fn,fm->fnm', X, np.conj(X), optimize=True)


def srp_map(X, A, phat=False):
    """
    beamforming, SRP-map via hi-mem computation, ca 1% C-time of pyroomacoustics (10% w einsum optimize=False)
    parameters:
        X (n_freq,n_channels,n_timesteps) OR (n_freq,n_channels,n_snapshots,n_timesteps)
        A
        phat (bool) use phase transform? (whitening, filtering "coloured reverberation" )
            if False, reduces to convenitonal beamforming
    output:
        P SRP power map
    """
    AHA = np.einsum('fnd,fmd->fnmd', np.conj(A), A)
    if phat:  # apply PHAseTransform
        X = X / np.abs(X + 1e-13)
    SCM = calc_scm(X.T)
    SRP = np.real(np.einsum('fnm,fnmd->d', SCM, AHA, optimize=True))
    # normalize by 1/2 (overlap?), n_freqs, n_mic_pairs
    SRP *= 2 / X.shape[0] / X.shape[1] / (X.shape[1] - 1)
    return SRP


def srpDOA(P, fvec, r_mic, c0=343., LookingDirections=None):
    if LookingDirections is None:
        LookingDirections = fib_sphere(2000)  # creates unit vectors in 3D directions, Nx3
    k_abs = 2 * np.pi * fvec / c0
    A = np.exp(1j * np.einsum('i,jk,dk->ijd', k_abs, r_mic.T, LookingDirections.T))  # time delay at microphones
    powermap = srp_map(P, A, phat=True).astype(np.float32)
    return LookingDirections[:,np.argmax(powermap)]


def find_MAP(x):
    try:
        mean_shift = MeanShift()
        mean_shift.fit(x)
        centers = mean_shift.cluster_centers_
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)

        best_center = (None, -np.inf)
        dens = kde.score_samples(centers)
        for c, d in zip(centers, dens):
            if d > best_center[1]:
                best_center = (c.copy(), d)

        dist_to_best = np.sum((x - best_center[0]) ** 2, axis=1)
        return np.argmin(dist_to_best)
    except:
        print('Mean shift failed')
        return 0


# @jit(nopython=True)
def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


# @jit(nopython=True)
def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)
    return c


# @jit(nopython=True)
def _disk_grid_fibonacci(n, r, c=(0, 0), z=None):
    """
    Get circular disk grid points
    Parameters
    ----------
    n : integer N, the number of points desired.
    r : float R, the radius of the disk.
    c : tuple of floats C(2), the coordinates of the center of the disk.
    z : float (optional), height of disk
    Returns
    -------
    cg :  real CG(2,N) or CG(3,N) if z != None, the grid points.
    """
    r0 = r / np.sqrt(float(n) - 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    gr = np.zeros(n)
    gt = np.zeros(n)
    for i in range(0, n):
        gr[i] = r0 * np.sqrt(i + 0.5)
        gt[i] = 2.0 * np.pi * float(i + 1) / phi

    if z is None:
        cg = np.zeros((3, n))
    else:
        cg = np.zeros((2, n))

    for i in range(0, n):
        cg[0, i] = c[0] + gr[i] * np.cos(gt[i])
        cg[1, i] = c[1] + gr[i] * np.sin(gt[i])
        if z != None:
            cg[2, i] = z
    return cg


# @jit(nopython=True)
def propagation_matmul(H, x):
    # return np.einsum('ijk, ik -> ij', H, x)
    return H @ x


def fib_sphere(num_points, radius=1.):
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)

    x_batch = np.dot(radius, x)
    y_batch = np.dot(radius, y)
    z_batch = np.dot(radius, z)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 3)
    # plt.show()
    return np.asarray([x_batch, y_batch, z_batch])


def wavenumber(f=1000, c=343):
    omega = 2 * np.pi * f  # angular frequency
    k = omega / c  # wavenumber
    return k


def SNRScale(sig, snrdB=40):
    ndim = sig.ndim
    if ndim >= 2:
        dims = (-2, -1)
    else:
        dims = -1
    mean = np.mean(sig, axis=dims)
    # remove DC
    if ndim > 2:
        sig_zero_mean = sig - mean[..., np.newaxis, np.newaxis]
    else:
        sig_zero_mean = sig - mean[..., np.newaxis]

    var = np.var(sig_zero_mean, axis=dims)
    if ndim >= 2:
        psig = var[..., np.newaxis, np.newaxis]
    else:
        psig = var[..., np.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    return psig / snr_lin


def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from signal
    ndim = sig.ndim
    if ndim >= 2:
        dims = (-2, -1)
    else:
        dims = -1
    mean = np.mean(sig, axis=dims)
    # remove DC
    if ndim > 2:
        sig_zero_mean = sig - mean[..., np.newaxis, np.newaxis]
    else:
        sig_zero_mean = sig - mean[..., np.newaxis]

    var = np.var(sig_zero_mean, axis=dims)
    if ndim >= 2:
        psig = var[..., np.newaxis, np.newaxis]
    else:
        psig = var[..., np.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        noise = np.sqrt(pnoise) * np.random.randn(sig.shape)
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        noise = real_noise + 1j * imag_noise
        noise_mag = np.sqrt(pnoise) * np.abs(noise)
        noise = noise_mag * np.exp(1j * np.angle(noise))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise


def get_spherical_array(n_mics, radius, add_interior_points=True, reference_grid=None):
    if not add_interior_points:
        return fib_sphere(n_mics, radius)
    else:
        assert reference_grid is not None
        rng = np.random.RandomState(1234)
        grid = fib_sphere(n_mics, radius)
        npoints = 5
        x_ref, y_ref, z_ref = reference_grid
        # number of interior points for zero-cross of bessel functions
        mask = np.argwhere(x_ref.ravel() ** 2 + y_ref.ravel() ** 2 <= radius ** 2)
        interp_ind = rng.choice(mask.shape[0], size=npoints, replace=False)
        interp_ind = np.squeeze(mask[interp_ind])
        grid = np.concatenate((grid, reference_grid[:, interp_ind]), axis=-1)
        return grid


def FindInterpolationIndex(grid_ref, grid):
    rng = np.random.RandomState(1234)
    npoints = 5
    mu = grid.mean(axis=-1)[..., None]
    tempgrid = grid - mu
    radius = np.linalg.norm(tempgrid, axis=0).min()
    mask = np.argwhere(grid_ref[0].ravel() ** 2 + grid_ref[1].ravel() ** 2 <= radius ** 2)
    index = rng.choice(mask.shape[0], size=npoints, replace=False)
    index = np.squeeze(mask[index])
    return index


def FindExtrapolationIndex(grid_ref, grid):
    rng = np.random.RandomState(1234)
    npoints = 5
    mu = grid.mean(axis=-1)[..., None]
    tempgrid = grid - mu
    radius = np.linalg.norm(tempgrid, axis=0).min()
    mask = np.argwhere(grid_ref[0].ravel() ** 2 + grid_ref[1].ravel() ** 2 > radius ** 2)
    index = rng.choice(mask.shape[0], size=npoints, replace=False)
    index = np.squeeze(mask[index])
    return index


def ConcatMeasurements(data1, data2, concatindex):
    if data1.ndim > 1:
        m, n = data1.shape
    else:
        m = len(data1)
        n = 1
    if m < n:
        data = np.concatenate((data1, data2[:, concatindex]), axis=-1)
    else:
        data = np.concatenate((data1, data2[concatindex]), axis=-1)
    return data


# grid = grids_sphere - grids_sphere.mean(axis = -1)[..., None]
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(grid[0],grid[1],grid[2])
# fig.show()
def distance_between(s, r):
    """Distance of all combinations of points in s and r.
    Parameters
    ----------
    s : ndarray, (3, ns)
    r : ndarray, (3, nr)
    Returns
    -------
    ndarray, (nr, ns)
        Distances between points
    """
    return np.linalg.norm(s[:, None, :] - r[:, :, None], axis=0)


def plane_waves(n0, k, grid, orthonormal=True):
    """
    x0 : (3,) array_like
        Position of plane wave.
    n0 : (3,) array_like
        Normal vector (direction) of plane wave.
    grid : triple of array_like
        The grid that is used for the sound field calculations.
    """

    # n0 = n0 / np.linalg.norm(n0, axis = -1)[..., None]

    P = np.exp(-1j * k * grid.T @ n0)
    if orthonormal:
        P /= np.sqrt(n0.shape[-1])
    return P

def point_sources(x0, k, grid, orthonormal=True, peturbate_phase = False):
    """
    x0 : (3, nsources) array_like
        Position of point source(s).
    k : (3, nfreqs) array_like
        Wavenumber.
    grid : (3, Nmicrophones) array_like
        The grid where the wave is evaluated.
    """

    # n0 = n0 / np.linalg.norm(n0, axis = -1)[..., None]
    if x0.ndim == 1:
        x0 = x0[:, None]
    r = np.linalg.norm(grid - x0, axis = 0)
    if peturbate_phase:
        phi = np.random.uniform(0., 2*np.pi,size = (k.shape[0], r.shape[0]))
    else:
        phi = 0.
    P = np.exp(-1j * k[:, None] * r + phi) / (4*np.pi*r)
    if orthonormal:
        P /= np.sqrt(x0.shape[-1])
    return P

def get_random_np_boolean_mask(n_true_elements, total_n_elements):
    assert total_n_elements >= n_true_elements
    a = np.zeros(total_n_elements, dtype=int)
    a[:n_true_elements] = 1
    np.random.shuffle(a)
    return a.astype(bool)


def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim=None, tex=False, cmap=None, normalise=True,
            colorbar=False, cbar_label='', cbar_loc='bottom'):
    """
    Plot spatial soundfield normalised amplitude
    --------------------------------------------
    Args:
        P : Pressure in meshgrid [X,Y]
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)
    """
    # plot_settings()

    N_interp = 1500
    if normalise:
        Pvec = P / np.max(abs(P))
    else:
        Pvec = P
    res = complex(0, N_interp)
    Xc, Yc = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    Pmesh = griddata(points, Pvec, (Xc, Yc), method='cubic', rescale=True)
    if cmap is None:
        cmap = 'coolwarm'
    if f is None:
        f = ''
    # P = P / np.max(abs(P))
    X = Xc.flatten()
    Y = Yc.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * X.ptp() / Pmesh.size
    dy = 0.5 * Y.ptp() / Pmesh.size
    if ax is None:
        _, ax = plt.subplots()  # create figure and axes
    im = ax.imshow(Pmesh.real, cmap=cmap, origin='upper',
                   extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    if clim is not None:
        lm1, lm2 = clim
        im.set_clim(lm1, lm2)
    if colorbar:
        if cbar_loc != 'bottom':
            shrink = 1.
            orientation = 'vertical'
        else:
            shrink = 1.
            orientation = 'horizontal'

        cbar = plt.colorbar(im, ax=ax, location=cbar_loc,
                            shrink=shrink)
        # cbar.ax.get_yaxis().labelpad = 15
        titlesize = int(1. * mpl.rcParams['axes.titlesize'])
        # cbar.ax.set_title(cbar_label, fontsize = titlesize)
        cbar.set_label(cbar_label, fontsize=titlesize)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name + ' - f : {} Hz'.format(f))
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax, im


def plot_settings():
    width = 6.694

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    }

    mpl.rcParams.update(tex_fonts)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    # plt.rcParams["figure.figsize"] = (6.694, 5)
    plt.rcParams['figure.constrained_layout.use'] = True
    return width


def array_to_cmplx(array):
    return array[..., 0] + 1j * array[..., 1]


def sample_unit_sphere(n_samples=2000):
    grid = fib_sphere(int(n_samples), 1)
    return grid
