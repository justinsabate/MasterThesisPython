import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm, jv, hankel2, hankel1
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib as mpl
from sklearn.linear_model import RidgeCV
from math import factorial
from scipy.special import legendre
import cvxpy as cp
import time
import math
from code_SH.spaudio_Utils import magls_bin


def get_eigenmike_grid(plot=False):
    # mpl.use('TkAgg')
    eigenmike_raw = {
        # colatitude, azimuth, radius
        # (degrees, degrees, meters)
        "1": [69, 0, 0.042],
        "2": [90, 32, 0.042],
        "3": [111, 0, 0.042],
        "4": [90, 328, 0.042],
        "5": [32, 0, 0.042],
        "6": [55, 45, 0.042],
        "7": [90, 69, 0.042],
        "8": [125, 45, 0.042],
        "9": [148, 0, 0.042],
        "10": [125, 315, 0.042],
        "11": [90, 291, 0.042],
        "12": [55, 315, 0.042],
        "13": [21, 91, 0.042],
        "14": [58, 90, 0.042],
        "15": [121, 90, 0.042],
        "16": [159, 89, 0.042],
        "17": [69, 180, 0.042],
        "18": [90, 212, 0.042],
        "19": [111, 180, 0.042],
        "20": [90, 148, 0.042],
        "21": [32, 180, 0.042],
        "22": [55, 225, 0.042],
        "23": [90, 249, 0.042],
        "24": [125, 225, 0.042],
        "25": [148, 180, 0.042],
        "26": [125, 135, 0.042],
        "27": [90, 111, 0.042],
        "28": [55, 135, 0.042],
        "29": [21, 269, 0.042],
        "30": [58, 270, 0.042],
        "31": [122, 270, 0.042],
        "32": [159, 271, 0.042],
    }
    temp = []
    for key in eigenmike_raw.keys():
        temp.append(eigenmike_raw[key])

    positions = np.array(temp).T
    # r_eig, phi_eig, theta_eig  = positions
    phi_eig, theta_eig, r_eig = positions

    theta_eig = np.deg2rad(theta_eig)
    phi_eig = np.deg2rad(phi_eig)
    x_eig, y_eig, z_eig = sph2cart(theta_eig, phi_eig, r_eig)

    if plot:
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        x_temp = (r_eig[0] - 0.1 * r_eig[0]) * np.cos(u) * np.sin(v)
        y_temp = (r_eig[0] - 0.1 * r_eig[0]) * np.sin(u) * np.sin(v)
        z_temp = (r_eig[0] - 0.1 * r_eig[0]) * np.cos(v)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_wireframe(x_temp, y_temp, z_temp,
                          rstride=2, cstride=2, color='white')
        ax.scatter(x_eig, y_eig, z_eig, s=20, color='k')
        set_aspect_equal(ax)
        # ax.view_init(30, 30)
        n = np.arange(1, 33)
        for i, txt in enumerate(n):
            ax.text(x_eig[i], y_eig[i], z_eig[i], '%s' % (str(txt)), size=10, zorder=1, color='k')

        fig.show()
    return np.array([x_eig, y_eig, z_eig])


def set_aspect_equal(ax):
    """
    Fix the 3D graph to have similar scale on all the axes.
    Call this after you do all the plot3D, but before show
    """
    X = ax.get_xlim3d()
    Y = ax.get_ylim3d()
    Z = ax.get_zlim3d()
    a = [X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]]
    b = np.amax(a)
    ax.set_xlim3d(X[0] - (b - a[0]) / 2, X[1] + (b - a[0]) / 2)
    ax.set_ylim3d(Y[0] - (b - a[1]) / 2, Y[1] + (b - a[1]) / 2)
    ax.set_zlim3d(Z[0] - (b - a[2]) / 2, Z[1] + (b - a[2]) / 2)
    ax.set_box_aspect(aspect=(1, 1, 1))


def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.
    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates
    Returns
    -------
    theta : float or `numpy.ndarray`
            Azimuth angle in radians
    phi : float or `numpy.ndarray`
            Colatitude angle in radians (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi, r


def sph2cart(azimuth, colatitude, r):
    """Spherical to cartesian coordinate transform.
    Parameters
    ----------
    azimuth : float or array_like
            Azimuth angle in radiants
    colatitude : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(azimuth) * np.sin(colatitude)
    y = r * np.sin(azimuth) * np.sin(colatitude)
    z = r * np.cos(colatitude)
    return x, y, z


# Forked from: https://github.com/AppliedAcousticsChalmers/sound_field_analysis-py
def mnArrays(nMax):
    """Generate degrees n and orders m up to nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    Returns
    -------
    m : (int), array_like
        0, -1, 0, 1, -2, -1, 0, 1, 2, ... , -nMax ..., nMax
    n : (int), array_like
        0, 1, 1, 1, 2, 2, 2, 2, 2, ... nMax, nMax, nMax
    """
    # Degree n = 0, 1, 1, 1, 2, 2, 2, 2, 2, ...
    degs = np.arange(nMax + 1)
    n = np.repeat(degs, degs * 2 + 1)

    # Order m = 0, -1, 0, 1, -2, -1, 0, 1, 2, ...
    # http://oeis.org/A196199
    elementNumber = np.arange((nMax + 1) ** 2) + 1
    t = np.floor(np.sqrt(elementNumber - 1)).astype(int)
    m = elementNumber - t * t - t - 1

    return m, n


def sph_harm_all(nMax, az, co, kind="complex"):
    """Compute all spherical harmonic coefficients up to degree nMax.
    Parameters
    ----------
    nMax : (int)
        Maximum degree of coefficients to be returned. n >= 0
    az: (float), array_like
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float), array_like
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    Returns
    -------
    y_mn : (complex float) or (float), array_like
        Spherical harmonics of degrees n [0 ... nMax] and all corresponding
        orders m [-n ... n], sampled at [az, co]. dim1 corresponds to az/co
        pairs, dim2 to oder/degree (m, n) pairs like 0/0, -1/1, 0/1, 1/1,
        -2/2, -1/2 ...
    """
    m, n = mnArrays(nMax)
    mA, azA = np.meshgrid(m, az)
    nA, coA = np.meshgrid(n, co)
    return sph_harmonics(mA, nA, azA, coA, kind=kind)


def sph_harmonics(m, n, az, co, kind="complex"):
    """Compute spherical harmonics.
    Parameters
    ----------
    m : (int)
        Order of the spherical harmonic. abs(m) <= n
    n : (int)
        Degree of the harmonic, sometimes called l. n >= 0
    az : (float)
        Azimuthal (longitudinal) coordinate [0, 2pi], also called Theta.
    co : (float)
        Polar (colatitudinal) coordinate [0, pi], also called Phi.
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type according to complex [7]_ or
        real definition [8]_ [Default: 'complex']
    Returns
    -------
    y_mn : (complex float) or (float)
        Spherical harmonic of order m and degree n, sampled at theta = az,
        phi = co
    References
    ----------
    .. [7] `scipy.special.sph_harm()`
    .. [8] Zotter, F. (2009). Analysis and Synthesis of Sound-Radiation with
        Spherical Arrays University of Music and Performing Arts Graz, Austria,
        192 pages.
    """
    # SAFETY CHECKS
    kind = kind.lower()
    if kind not in ["complex", "real"]:
        raise ValueError("Invalid kind: Choose either complex or real.")
    m = np.atleast_1d(m)

    Y = sph_harm(m, n, az, co)
    if kind == "complex":
        return Y
    else:  # kind == 'real'
        mg0 = m > 0
        ml0 = m < 0
        Y[mg0] = np.float_power(-1.0, m)[mg0] * np.sqrt(2) * np.real(Y[mg0])
        Y[ml0] = np.sqrt(2) * np.imag(Y[ml0])
        return np.real(Y)


def spatialFT(data, position_grid, grid_type='cart', order_max=10, kind="complex",
              spherical_harmonic_bases=None, weight=None,
              leastsq_fit=False, regularised_lstsq_fit=False, MLS=False, fs=None, NFFT = 4096):
    """Perform spatial Fourier transform.
    Parameters
    ----------
    data : array_like
        Data to be transformed, with signals in rows and frequency bins in
        columns
    position_grid : array_like cartesian coordinates of spatial sampling points [3, Npoints]
    grid_type : {'cart','sphe'} cartesian or spherical to convert or not according to type
    order_max : int, optional
        Maximum transform order [Default: 10]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]
    Returns
    -------
    Pnm : array_like
        Spatial Fourier Coefficients with nm coeffs in rows and FFT bins in
        columns
    Notes
    -----
    In case no weights in spatial sampling grid are given, the pseudo inverse
    of the SH bases is computed according to Eq. 3.34 in [5]_.
    References
    ----------
    .. [5] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
        (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg,
        2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8
        :param kind:
        :param grid_type:
    """
    # Justin adding
    if grid_type == 'cart':
        azi, elev, r = cart2sph(position_grid[0], position_grid[1], position_grid[2])
    elif grid_type == 'sphe':
        azi, elev, r = position_grid[0], position_grid[1], position_grid[2]

    if spherical_harmonic_bases is None:
        spherical_harmonic_bases = sph_harm_all(
            order_max, azi, elev, kind=kind
        )
    if leastsq_fit:
        return np.linalg.lstsq(spherical_harmonic_bases, data, rcond=None)[0]  # , spherical_harmonic_bases,azi, elev

    # Justin custom solver, magnitude least squares
    if MLS:
        ''' Example adapted from https://www.cvxpy.org/examples/basic/least_squares.html'''
        'Not possible to give the code any hint in the beginning, but the warm start is using the previous result, ' \
        'it might be already what is mentioned in paper 39 '

        print('Using Magnitude least squares')



        '''Try 1 : for loop over frequencies -> impossible because way too long, even for one only frequency bin'''
        # Problem data.
        # Y will be the spherical basis
        # Y = spherical_harmonic_bases
        #
        # # h will be the hrtf set
        # h = data  # otherwise cannot handle complex values
        # # frequency dependent factor
        # i_fc = 1  # 256  # corresponds to 2kHz with fs = 32kHz and Nfft = 4096 #TODO(set it to 256 if running with NFFT = 4096)
        # Lam = np.zeros(np.shape(h)[1])
        # Lam[0:i_fc] = np.ones(i_fc)
        #
        # # has to be solved for each frequency bin
        # solution = np.zeros((np.shape(h)[0], np.shape(Y)[1]))
        # for i in range(len(Lam)):
        #     # Define and solve the CVXPY problem.
        #     if i >= 1: # could create 2 variables and the condition abs(x) == y ?
        #         w = cp.Variable(shape=np.shape(Y)[1], value=w.value)
        #     else:
        #         w = cp.Variable(shape=np.shape(Y)[1])
        #
        #     # trying to initialize -> no difference in running time
        #     # if i != 0 :
        #     #     w.value = solution[i-1]
        #
        #     '''Try Antonio's idea : split the problem into 2'''
        #     print('lambda = '+str(Lam[i]))
        #     # if Lam[i] == 0:
        #     #     cost = cp.sum_squares((Y @ w) - cp.abs(h[:, i]))
        #     # else:
        #     #     cost = cp.sum_squares(Y @ w - h[:, i])
        #
        #     # Lambda = i
        #     # coeff1 = Y @ w
        #     # coeff2 = h[:, i]
        #     #
        #     # coeff1 = cp.sqrt(cp.real(coeff1)**2+cp.imag(coeff1)**2)
        #     # coeff2 = cp.sqrt(cp.real(coeff2) ** 2 + cp.imag(coeff2) ** 2)
        #
        #     # cost = Lam[i] * cp.sum_squares(Y @ w - h[:, i]) + (1 - Lam[i]) * cp.sum_squares(coeff1 - coeff2)
        #
        #
        #     # cost = Lam[i] * cp.sum_squares(Y @ w - h[:, i]) + (1 - Lam[i]) * cp.sum_squares(
        #     #     cp.norm1(Y @ w) - cp.abs(h[:, i]))  # not taking the absolute value of x here but should take it
        #
        #     cost = Lam[i] * cp.sum_squares(Y @ w - h[:, i]) + (1 - Lam[i]) * cp.sum_squares(
        #         cp.norm(Y @ w) - cp.norm(h[:, i]))  # not taking the absolute value of x here but should take it
        #
        #     prob = cp.Problem(cp.Minimize(cost))
        #     print("Solving the inverse problem nb " + str(i))
        #
        #     # trying to time the operation
        #     start = time.time()
        #     prob.solve()
        #     end = time.time()
        #     delta = end - start
        #     print("solving took " + str(delta) + " s")
        #     solution[i] = w.value
        #     # TOO LONG TO RUN + not running if abs value still here

        '''Try 2 : 3D matrices -> impossible because cost function cannot be a matrix'''
        # # Problem data.
        # # Y will be the spherical basis
        # Y = spherical_harmonic_bases
        # # h will be the hrtf set
        # h = data  # otherwise cannot handle complex values
        # # frequency dependent factor
        # i_fc = 1  # 256  # corresponds to 2kHz with fs = 32kHz and Nfft = 4096
        # Lambda = np.zeros(np.shape(h)[1])
        # Lambda[0:i_fc] = np.ones(i_fc)

        # # other try with 3D matrices
        # w = cp.Variable((np.shape(Y)[1], np.shape(h)[1]))
        # cost = Lambda * cp.sum_squares(Y @ w - h) + (1 - Lambda) * cp.sum_squares(
        #     cp.abs(Y) @ w - cp.abs(h))  # not taking the absolute value of x here but should take it
        # constraints = [0 <= x, x <= 1]

        # # cannot have a cost matrix, but a cost should be a scalar
        # solution = np.zeros(np.shape(h))
        # for i in range(len(Lambda)):
        #     prob = cp.Problem(cp.Minimize(cost[i]))
        #     prob.solve()
        #     solution[i] = w.value

        '''Try 3 mapping complex to non complex values'''  # Much faster if still no absolute value in the formula
        # n_harm = spherical_harmonic_bases.shape[-1]
        # Y = np.real(stack_real_imag_Y(spherical_harmonic_bases))
        # h = np.concatenate((data.real, data.imag))
        # i_fc = 256  # corresponds to 2kHz with fs = 32kHz and Nfft = 4096
        # Lambda = np.zeros(np.shape(h)[1])
        # Lambda[0:i_fc] = np.ones(i_fc)
        # solution = np.zeros((np.shape(data)[0], np.shape(spherical_harmonic_bases)[1]), dtype=np.complex_)
        # for i in range(len(Lambda)):
        #     # Define and solve the CVXPY problem.
        #     w = cp.Variable(np.shape(Y)[1])  # could create 2 variables and the condition abs(x) == y ?
        #     cost = Lambda[i] * cp.sum_squares(Y @ w - h[:, i]) + (1 - Lambda[i]) * cp.sum_squares(
        #         cp.abs(Y @ w) - cp.abs(h[:, i]))  # TODO(might be wrong, replacing the abs by non abs and concatenating the complex into real values to make it dcp)
        #     prob = cp.Problem(cp.Minimize(cost))
        #     # The optimal objective value is returned by `prob.solve()`.
        #     print("Solving the inverse problem nb " + str(i))
        #     prob.solve()
        #     solution[i] = w.value[:n_harm]+1j*w.value[n_harm:]

        # 'Try 4 : solving the problem before optimization in the paper with 2 variables'
        # print('Using Magnitude least squares')
        # # Problem data.
        # # Y will be the spherical basis
        # Y = spherical_harmonic_bases
        #
        # # h will be the hrtf set
        # h = data  # otherwise cannot handle complex values
        # # frequency dependent factor
        # i_fc = 1  # 256  # corresponds to 2kHz with fs = 32kHz and Nfft = 4096 #TODO(set it to 256 if running with NFFT = 4096)
        # Lam = np.zeros(np.shape(h)[1])
        # Lam[0:i_fc] = np.ones(i_fc)
        #
        # # has to be solved for each frequency bin
        # solution = np.zeros((np.shape(h)[0], np.shape(Y)[1]))
        #
        #
        # # has to make it i dependent with the lambda
        # for i in range(len(Lam)):
        #     # Define and solve the CVXPY problem.
        #     w = cp.Variable(shape=np.shape(Y)[1])
        #     p = cp.Variable(shape=np.shape(h)[0])
        #
        #     print('lambda = '+str(Lam[i]))
        #
        #     M = cp.diag(np.abs(h[:, i]))
        #
        #     cost = cp.sum_squares(Y@w-M@p)
        #     constraint = []
        #     for j in range(np.size(p)):
        #         constraint.append(cp.norm2(p[j]-1) == 0)
        #     objective = cp.Minimize(cost)
        #     prob = cp.Problem(objective, constraint)
        #     print("Solving the inverse problem nb " + str(i))
        #
        #     # trying to time the operation
        #     start = time.time()
        #     prob.solve()
        #     end = time.time()
        #     delta = end - start
        #     print("solving took " + str(delta) + " s")
        #     solution[i] = w.value
        #     # TOO LONG TO RUN + not running if abs value still here
        #
        ''' Try 5 new library'''

        # spherical_harmonic_bases
        # data

        grid = position_grid # TODO check if cart or sph needed
        fs = fs
        HRIR_L = data[0,:,:]
        HRIR_R = data[1,:,:]
        azi = azi
        zen = elev
        solution = magls_bin(hrirs=data, N_sph=order_max, f_trans=2000,fs = fs, azi = azi,elev = elev,gridpoints = position_grid, Nfft=NFFT, basis=spherical_harmonic_bases)

        return solution  # matrix multiplication with the HRTF (data in the frequency domain) already done in magls_bin

    if regularised_lstsq_fit:
        # i assume this creates the condition/regularization
        reg = RidgeCV(cv=5, alphas=np.geomspace(1, 1e-7, 50), fit_intercept=True)
        # reshaping the data
        data = np.concatenate((data.real, data.imag))
        # getting the order
        n_harm = spherical_harmonic_bases.shape[-1]
        # getting the basis
        spherical_harmonic_bases = stack_real_imag_Y(spherical_harmonic_bases)
        # processing the regularized least square
        reg.fit(spherical_harmonic_bases, data)
        # shaping the data
        q = reg.coef_[:, :n_harm] + 1j * reg.coef_[:, n_harm:]
        return q.T
    else:
        if weight is None:
            # calculate pseudo inverse in case no spatial sampling point weights
            # are given
            spherical_harmonics_weighted = np.linalg.pinv(spherical_harmonic_bases)
        else:
            # apply spatial sampling point weights in case they are given
            spherical_harmonics_weighted = np.conj(spherical_harmonic_bases).T * (
                    4 * np.pi * weight
            )

        return np.dot(spherical_harmonics_weighted, data)


def ispatialFT(
        spherical_coefficients,
        position_grid,
        order_max=None,
        kind="complex",
        spherical_harmonic_bases=None,
):
    """Perform inverse spatial Fourier transform.
    Parameters
    ----------
    spherical_coefficients : array_like
        Spatial Fourier coefficients with columns representing frequency bins
    position_grid : array_like or io.SphericalGrid
        Azimuth/Colatitude angles of spherical coefficients
    order_max : int, optional
        Maximum transform order [Default: highest available order]
    kind : {'complex', 'real'}, optional
        Spherical harmonic coefficients data type [Default: 'complex']
    spherical_harmonic_bases : array_like, optional
        Spherical harmonic base coefficients (not yet weighted by spatial
        sampling grid) [Default: None]
    Returns
    -------
    P : array_like
        Sound pressures with frequency bins in columns and angles in rows
    TODO
    ----
    Check `spherical_coefficients` and `spherical_harmonic_bases` length
    correspond with `order_max`
    """
    spherical_coefficients = np.atleast_2d(spherical_coefficients)
    number_of_coefficients = spherical_coefficients.shape[0]

    azi, elev, r = cart2sph(position_grid[0], position_grid[1], position_grid[2])

    # TODO: Check `spherical_coefficients` and `spherical_harmonic_bases` length
    #  correspond with `order_max`
    if order_max is None:
        order_max = int(np.sqrt(number_of_coefficients) - 1)

    # Re-generate spherical harmonic bases if they were not provided or their
    # order is too small
    if (
            spherical_harmonic_bases is None
            or spherical_harmonic_bases.shape[1] < number_of_coefficients
            or spherical_harmonic_bases.shape[1] != azi.size
    ):
        spherical_harmonic_bases = sph_harm_all(
            order_max, azi, elev, kind=kind
        )

    return np.dot(spherical_harmonic_bases, spherical_coefficients)


def stack_real_imag_Y(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def extrapolate(
        spherical_coefficients,
        extrap_grid,
        measurement_grid,
        k,
        order_max=None,
        spherical_harmonic_bases=None):
    r"""
    Extrapolate the sound pressure to positions beyond the radius of the measurement sphere
    and assuming the source is also beyond the spherical measurement radius as described by [1].

    .. math:: p(k, r^{'}, \theta^{'}, \phi^{'}) =
    .. math:: \sum_{n=0}^{\infty} \sum_{m=-n}^{n} \frac{j_n(kr^{'})}{j_n(kr)} P_{nm}(k, r) Y_{n}^{m}(\theta^{'}, \phi^{'})

    Parameters
    ----------
    spherical_coefficients : Measured or simulated spherical Fourier coefficients (obtained
                             with spatialFT) [nHarmonics x nFreqBins]
    extrap_grid : Grid to extrapolate sound pressure to in cartesian coords [3 x n_new_points]
    measurement_grid : Grid of measurement array [3 x n_points]
    order_max: (optional) Maximum spherical harmonic order. If not given, calculated as
                square root of (number of coefficients - 1) (int)
    spherical_harmonic_bases: (optional) spherical harmonics pre-calculated [n_points x nHarmonics]

    Returns
    -------
    Pressure calculated at positions given by extrap_grid and as a function of frequency
    [n_new_points x nFreqBins]

    [1] Rafaely, Boaz. Fundamentals of spherical array processing. Vol. 16. Springer, 2018. pp. 47, eq. 2.47
    """

    spherical_coefficients = np.atleast_2d(spherical_coefficients)
    number_of_coefficients = spherical_coefficients.shape[0]

    azi, elev, r = cart2sph(extrap_grid[0], extrap_grid[1], extrap_grid[2])
    azi_m, elev_m, rm = cart2sph(measurement_grid[0], measurement_grid[1], measurement_grid[2])

    if order_max is None:
        order_max = int(np.sqrt(number_of_coefficients) - 1)
    ka = k * rm.mean()[..., None, None]
    kr = k * r[..., None]

    if np.any(kr[:, 0] == 0):
        kr[:, 0] = kr[:, 1]
    if np.any(ka[:, 0] == 0):
        ka[:, 0] = ka[:, 1]

    # Re-generate spherical harmonic bases if they were not provided or their
    # order is too small
    if (
            spherical_harmonic_bases is None
            or spherical_harmonic_bases.shape[1] < number_of_coefficients
            or spherical_harmonic_bases.shape[1] != azi.size
    ):
        spherical_harmonic_bases = sph_harm_all(
            order_max, azi, elev, kind="complex"
        )
    radial_functions = np.zeros((kr.shape[0], kr.shape[1], spherical_harmonic_bases.shape[-1]))
    nold = 0

    for n in range(0, order_max + 1):
        jn = spbessel(n, kr) / spbessel(n, ka)
        nnew = nold + 2 * n + 1
        radial_functions[:, :, nold:nnew] = jn[..., None]
        nold = nnew

    return np.einsum('rfn, nf, rn -> rf', radial_functions, spherical_coefficients, spherical_harmonic_bases)


def weights(N, kr, setup):
    r"""Radial weighing functions.
    Computes the radial weighting functions for different array types
    (cf. eq.(2.62), Rafaely 2015).
    For instance for an rigid array
    .. math::
        b_n(kr) = j_n(kr) - \frac{j_n^\prime(kr)}{h_n^{(2)\prime}(kr)}h_n^{(2)}(kr)
    Parameters
    ----------
    N : int
        Maximum order.
    kr : (M,) array_like
        Wavenumber * radius.
    setup : {'open', 'card', 'rigid'}
        Array configuration (open, cardioids, rigid).
    Returns
    -------
    bn : (M, N+1) numpy.ndarray
        Radial weights for all orders up to N and the given wavenumbers.
    """
    kr = np.atleast_1d(kr)
    n = np.arange(N + 1)
    bns = np.zeros((len(kr), N + 1), dtype=complex)
    for i, x in enumerate(kr):
        jn = spherical_jn(n, x)
        if setup == 'open':
            bn = jn
        elif setup == 'card':
            bn = jn - 1j * spherical_jn(n, x, derivative=True)
        elif setup == 'rigid':
            if x == 0:
                # hn(x)/hn'(x) -> 0 for x -> 0
                bn = jn
            else:
                jnd = spherical_jn(n, x, derivative=True)
                hn = jn - 1j * spherical_yn(n, x)
                hnd = jnd - 1j * spherical_yn(n, x, derivative=True)
                bn = jn - jnd / hnd * hn
        else:
            raise ValueError('setup must be either: open, card or rigid')
        bns[i, :] = bn
    return np.squeeze(bns)


def sound_field_separation(spherical_coefficients,
                           extrap_grid,
                           measurement_grid,
                           k,
                           c=343.,
                           rho=1.2,
                           order_max=None,
                           return_particle_velocity=False,
                           Nmax=4
                           ):
    spherical_coefficients = np.atleast_2d(spherical_coefficients)
    number_of_coefficients = spherical_coefficients.shape[0]

    azi, elev, r = cart2sph(extrap_grid[0], extrap_grid[1], extrap_grid[2])
    azi_m, elev_m, rm = cart2sph(measurement_grid[0], measurement_grid[1], measurement_grid[2])
    omega = k * c

    if order_max is None:
        order_max = int(np.sqrt(number_of_coefficients) - 1)
    ka = k * rm.mean()[..., None, None]
    kr = k * r[..., None]

    if np.any(kr[:, 0] == 0):
        kr[:, 0] = kr[:, 1]
    if np.any(ka[:, 0] == 0):
        ka[:, 0] = ka[:, 1]

    jn_prime_ka = -get_bessel_to_order_n(order_max, ka, derivative=True)  # typo in EFrens paper
    hn2_prime_ka = get_hankel2_to_order_n(order_max, ka, derivative=True)

    jn_kr = get_bessel_to_order_n(order_max, kr, derivative=False)
    hn2_kr = get_hankel2_to_order_n(order_max, kr, derivative=False)

    Cm = 1j * np.einsum('rf, rfn, nf -> nf', ka ** 2, jn_prime_ka, spherical_coefficients)
    Bm = 1j * np.einsum('rf, rfn, nf -> nf', ka ** 2, hn2_prime_ka, spherical_coefficients)

    spherical_harmonic_bases = sph_harm_all(
        Nmax, azi, elev, kind="complex")

    p_inc = np.einsum('rfn, nf, rn -> rf', jn_kr, Bm, spherical_harmonic_bases)
    p_scat = np.einsum('rfn, nf, rn -> rf', hn2_kr, Cm, spherical_harmonic_bases)
    if return_particle_velocity:
        djn_kr = get_bessel_to_order_n(Nmax, kr, derivative=True)
        dhn2_kr = get_hankel2_to_order_n(Nmax, kr, derivative=True)

        u_inc = -1 / (1j * omega * rho) * np.einsum('rfn, nf, rn -> rf', djn_kr, Bm, spherical_harmonic_bases)
        u_scat = -1 / (1j * omega * rho) * np.einsum('rfn, nf, rn -> rf', dhn2_kr, Cm, spherical_harmonic_bases)
        return p_inc, p_scat, u_inc, u_scat
    else:
        return p_inc, p_scat


def besselj(n, z):
    """Bessel function of first kind of order n at kr. Wraps
    `scipy.special.jn(n, z)`.
    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument
    Returns
    -------
    J : array_like
        Values of Bessel function of order n at position z
    """
    return jv(n, np.complex_(z))


def neumann(n, z):
    """Bessel function of second kind (Neumann / Weber function) of order n at
    kr. Implemented as `(hankel1(n, z) - besselj(n, z)) / 1j`.
    Parameters
    ----------
    n : array_like
        Order
    z: array_like
        Argument
    Returns
    -------
    Y : array_like
        Values of Hankel function of order n at position z
    """
    return (hankel1(n, z) - besselj(n, z)) / 1j


def spbessel(n, kr):
    """Spherical Bessel function (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    J : complex float
        Spherical Bessel
    """
    n, kr = scalar_broadcast_match(n, kr)

    if np.any(n < 0) | np.any(kr < 0) | np.any(np.mod(n, 1) != 0):
        J = np.zeros(kr.shape, dtype=np.complex_)

        kr_non_zero = kr != 0
        J[kr_non_zero] = np.lib.scimath.sqrt(np.pi / 2 / kr[kr_non_zero]) * besselj(
            n[kr_non_zero] + 0.5, kr[kr_non_zero]
        )
        J[np.logical_and(kr == 0, n == 0)] = 1
    else:
        J = spherical_jn(n.astype(int), kr)
    return np.squeeze(J)


def spneumann(n, kr):
    """Spherical Neumann (Bessel second kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    Yv : complex float
        Spherical Neumann (Bessel second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)

    if np.any(n < 0) | np.any(np.mod(n, 1) != 0):
        Yv = np.full(kr.shape, np.nan, dtype=np.complex_)

        kr_non_zero = kr != 0
        Yv[kr_non_zero] = np.lib.scimath.sqrt(np.pi / 2 / kr[kr_non_zero]) * neumann(
            n[kr_non_zero] + 0.5, kr[kr_non_zero]
        )
        Yv[kr < 0] = -Yv[kr < 0]
    else:
        Yv = spherical_yn(n.astype(np.int), kr)
        # return possible infs as nan to stay consistent
        Yv[np.isinf(Yv)] = np.nan
    return np.squeeze(Yv)


def sphankel1(n, kr):
    """Spherical Hankel (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    hn1 : complex float
        Spherical Hankel function hn (first kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    hn1 = np.full(n.shape, np.nan, dtype=np.complex_)
    kr_nonzero = kr != 0
    hn1[kr_nonzero] = (
            np.sqrt(np.pi / 2)
            / np.lib.scimath.sqrt(kr[kr_nonzero])
            * hankel1(n[kr_nonzero] + 0.5, kr[kr_nonzero])
    )
    return hn1


def sphankel2(n, kr):
    """Spherical Hankel (second kind) of order n at kr
    Parameters
    ----------
    n : array_like
       Order
    kr: array_like
       Argument
    Returns
    -------
    hn2 : complex float
       Spherical Hankel function hn (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    hn2 = np.full(n.shape, np.nan, dtype=np.complex_)
    kr_nonzero = kr != 0
    hn2[kr_nonzero] = (
            np.sqrt(np.pi / 2)
            / np.lib.scimath.sqrt(kr[kr_nonzero])
            * hankel2(n[kr_nonzero] + 0.5, kr[kr_nonzero])
    )
    return hn2


def dspbessel(n, kr):
    """Derivative of spherical Bessel (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    J' : complex float
        Derivative of spherical Bessel
    """
    return np.squeeze(
        (n * spbessel(n - 1, kr) - (n + 1) * spbessel(n + 1, kr)) / (2 * n + 1)
    )


def dspneumann(n, kr):
    """Derivative spherical Neumann (Bessel second kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    Yv' : complex float
        Derivative of spherical Neumann (Bessel second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    if np.any(n < 0) | np.any(np.mod(n, 1) != 0) | np.any(np.mod(kr, 1) != 0):
        return spneumann(n, kr) * n / kr - spneumann(n + 1, kr)
    else:
        return spherical_yn(
            n.astype(np.int), kr.astype(np.complex), derivative=True
        )


def dsphankel1(n, kr):
    """Derivative spherical Hankel (first kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    dhn1 : complex float
        Derivative of spherical Hankel function hn' (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    dhn1 = np.full(n.shape, np.nan, dtype=np.complex)
    kr_nonzero = kr != 0
    dhn1[kr_nonzero] = 0.5 * (
            sphankel1(n[kr_nonzero] - 1, kr[kr_nonzero])
            - sphankel1(n[kr_nonzero] + 1, kr[kr_nonzero])
            - sphankel1(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero]
    )
    return dhn1


def dsphankel2(n, kr):
    """Derivative spherical Hankel (second kind) of order n at kr.
    Parameters
    ----------
    n : array_like
        Order
    kr: array_like
        Argument
    Returns
    -------
    dhn2 : complex float
        Derivative of spherical Hankel function hn' (second kind)
    """
    n, kr = scalar_broadcast_match(n, kr)
    dhn2 = np.full(n.shape, np.nan, dtype=np.complex_)
    kr_nonzero = kr != 0
    dhn2[kr_nonzero] = 0.5 * (
            sphankel2(n[kr_nonzero] - 1, kr[kr_nonzero])
            - sphankel2(n[kr_nonzero] + 1, kr[kr_nonzero])
            - sphankel2(n[kr_nonzero], kr[kr_nonzero]) / kr[kr_nonzero]
    )
    return dhn2


def GreenNeumannESMMat(grid_src, grid_mic, Nmax, k, a):
    """ rigid sphere """
    az_src, elev_src, r_src = cart2sph(grid_src[0], grid_src[1], grid_src[2])
    az_mic, elev_mic, r_mic = cart2sph(grid_mic[0], grid_mic[1], grid_mic[2])
    # Build spherical harmonics and Hankel functions matrix for equiv sources
    ms, ns = mnArrays(Nmax)
    radial_functions = []
    for m, n in zip(ms, ns):
        hn_kr0 = sphankel2(n, k * r_src.mean(0))
        hn_ka_prime = sphankel2(n, k * r_mic.mean(0))
        radial_functions.append(hn_kr0 / hn_ka_prime)
    Ymn_src = sph_harm_all(Nmax, az_src, elev_src)

    radial_functions = np.array(radial_functions)
    # Build spherical harmonics matrix for microphones
    Ymn_mic = sph_harm_all(Nmax, az_mic, elev_mic)
    # Compute the green neumann function for each microphone/source group
    Gn = -1 / (k * a ** 2) * (Ymn_mic @ np.einsum('i, ij -> ij', radial_functions, Ymn_src.T))
    return Gn


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


def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim=None, tex=False, cmap=None, normalise=True,
            colorbar=False, cbar_label='', cbar_loc='bottom'):
    """

    Parameters
    ----------
    P
    x
    y
    f
    ax
    name
    save
    add_meas
    clim
    tex
    cmap
    normalise
    colorbar
    cbar_label
    cbar_loc

    Returns
    -------

    """

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
    im = ax.imshow(Pmesh.real.T, cmap=cmap, origin='lower',
                   extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    # ax.invert_xaxis()
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
        titlesize = mpl.rcParams['axes.titlesize']
        # cbar.ax.set_title(cbar_label, fontsize = titlesize)
        cbar.set_label(cbar_label, fontsize=titlesize)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name + ' - f : {:.2f} Hz'.format(f))
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax, im


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


def scalar_broadcast_match(a, b):
    """Returns arguments as np.array, if one is a scalar it will broadcast the
    other one's shape.
    """
    a, b = np.atleast_1d(a, b)
    if a.size == 1 and b.size != 1:
        a = np.broadcast_to(a, b.shape)
    elif b.size == 1 and a.size != 1:
        b = np.broadcast_to(b, a.shape)
    return a, b


def reference_grid(steps, xmin=-.7, xmax=.7, z=0):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones(X.shape)
    return np.array([X.ravel(), Y.ravel(), Z.ravel()])


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


def plot_array_pressure(p_array, array_grid, ax=None, plane=False, norm=None, z_label=False):
    if ax is None:
        if z_label:
            ax = plt.axes(projection='3d')
        else:
            ax = plt.axes()

    cmp = plt.get_cmap("RdBu")
    if norm is None:
        vmin = p_array.real.min()
        vmax = p_array.real.max()
    else:
        vmin, vmax = norm
    if z_label:
        sc = ax.scatter(array_grid[0], array_grid[1], array_grid[2], c=p_array.real,
                        cmap=cmp, alpha=1., s=10, vmin=vmin, vmax=vmax)
    else:
        sc = ax.scatter(array_grid[0], array_grid[1], c=p_array.real,
                        cmap=cmp, alpha=1., s=10, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if z_label:
        ax.set_zlabel('z [m]')
        ax.view_init(45, 45)

        if plane:
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_box_aspect((array_grid[0].max(), array_grid[1].max(), array_grid[2].max()))
    return ax, sc


def get_bessel_to_order_n(order_max, kr, derivative=False):
    if derivative:
        func = lambda n, kr: dspbessel(n, kr)
    else:
        func = lambda n, kr: spbessel(n, kr)
    nold = 0
    NMLocatorSize = (order_max + 1) ** 2
    besselfun = np.zeros((kr.shape[0], kr.shape[1], NMLocatorSize), dtype=complex)

    for n in range(0, order_max + 1):
        jn = func(n, kr)
        nnew = nold + 2 * n + 1
        besselfun[:, :, nold:nnew] = jn[..., None]
        nold = nnew
    return besselfun


def get_hankel2_to_order_n(order_max, kr, derivative=False):
    if derivative:
        func = lambda n, kr: dsphankel2(n, kr)
    else:
        func = lambda n, kr: sphankel2(n, kr)
    nold = 0
    NMLocatorSize = (order_max + 1) ** 2
    besselfun = np.zeros((kr.shape[0], kr.shape[1], NMLocatorSize), dtype=complex)

    for n in range(0, order_max + 1):
        jn = func(n, kr)
        nnew = nold + 2 * n + 1
        besselfun[:, :, nold:nnew] = jn[..., None]
        nold = nnew
    return besselfun


def bn_rigid_omni(n, kr, ka, normalize=False):
    """
    Radial function for scattering of rigid sphere from a point source
    Parameters
    ----------
    n: order of radial functions
    kr: Helmholtz number for mic radius (scalar or array-like)
    ka: Helmholtz number for sphere radius (scalar or array-like)

    Returns
    -------
    radial function for rigid sphere
    """
    if normalize:
        scale_factor = np.squeeze(4 * np.pi * 1j ** n)
    else:
        scale_factor = 1.
    kr, ka = scalar_broadcast_match(kr, ka)
    return scale_factor * (spbessel(n, kr) - (
            (dspbessel(n, ka) / dsphankel2(n, ka)) * sphankel2(n, kr)
    ))


def bn_open_omni(n, kr, normalize=False):
    if normalize:
        scale_factor = np.squeeze(4 * np.pi * 1j ** n)
    else:
        scale_factor = 1.

    return scale_factor * spherical_jn(n, kr)


def point_source_sphere_radial_filters(order_max,
                                       k,
                                       sphere_grid,
                                       source_loc,
                                       sphere_type='rigid',
                                       c=343.,
                                       delay=0.):
    NFFT = len(k)
    freqs = k * c / (2 * np.pi)
    omega = 2 * np.pi * freqs
    time_shift = np.exp(-1j * omega * delay)

    az_s, elev_s, r_s = cart2sph(source_loc[0], source_loc[1], source_loc[2])
    az_mic, elev_mic, r_mic = cart2sph(sphere_grid[0], sphere_grid[1], sphere_grid[2])

    ka = k * r_mic.mean(0)[None, None]  # scatter radius
    kr_source = k * r_s[..., None]

    if np.any(kr_source[:, 0] == 0):
        kr_source[:, 0] = kr_source[:, 1]
    if np.any(ka[:, 0] == 0):
        ka[:, 0] = ka[:, 1]

    NMLocatorSize = (order_max + 1) ** 2

    radial_filters = np.zeros([NMLocatorSize, NFFT], dtype=complex)
    for n in range(0, order_max + 1):
        if sphere_type == 'rigid':
            bn = bn_rigid_omni(n, ka, ka)
        else:  # open
            bn = bn_open_omni(n, ka)
        radial_filters[n] = (
                (4 * np.pi * -1j * k * time_shift)
                * sphankel2(n, kr_source)
                * bn
        )
    return radial_filters


def point_source_spherical_coefficients(order_max,
                                        k,
                                        source_loc,
                                        sphere_grid,
                                        sphere_type='rigid'
                                        ):
    NFFT = len(k)
    az_s, elev_s, _ = cart2sph(source_loc[0], source_loc[1], source_loc[2])

    radial_filters = point_source_sphere_radial_filters(order_max,
                                                        k,
                                                        sphere_grid,
                                                        source_loc,
                                                        sphere_type)
    NMLocatorSize = (order_max + 1) ** 2

    # SPATIAL FOURIER COEFFICIENTS
    Pnm = np.empty([NMLocatorSize, NFFT], dtype=complex)
    ctr = 0
    for n in range(0, order_max + 1):
        for m in range(-n, n + 1):
            Pnm[ctr] = (
                    np.conj(sph_harmonics(m, n, az_s, elev_s)) *
                    radial_filters[n]
            )
            ctr = ctr + 1
    return Pnm


def reverse_nm(SH_max_order):
    nm = (SH_max_order + 1) ** 2
    n_list = np.arange(0, SH_max_order + 1)
    nm_indexes_reversed = np.array([])
    for n in range(0, SH_max_order + 1):
        nm_indexes_reversed = np.concatenate((nm_indexes_reversed, np.flip(np.arange(n ** 2, (n + 1) ** 2))))
    return nm_indexes_reversed.astype(int)


def get_n_m(nm, Nmax):
    for n in range(0, Nmax + 1):
        if nm in range(n ** 2, (n + 1) ** 2):
            m = nm - n ** 2
            return n, m - n
    return None

