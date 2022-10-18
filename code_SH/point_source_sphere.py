from code_SH.SphHarmUtils import _disk_grid_fibonacci, cart2sph, \
    point_source_spherical_coefficients, \
                    plot_sf, plot_array_pressure, ispatialFT, extrapolate, dspbessel, dsphankel2, \
                    spbessel, sphankel2, sph_harm_all, get_eigenmike_grid
import numpy as np
from get_ico_coords import get_ico_coords
import matplotlib.pyplot as plt

def get_bessel_to_order_n(order_max, kr, derivative = False):
    if derivative:
        func = lambda n, kr : dspbessel(n, kr)
    else:
        func = lambda n, kr : spbessel(n, kr)
    nold = 0
    NMLocatorSize = (order_max + 1) ** 2
    besselfun = np.zeros((kr.shape[0], kr.shape[1], NMLocatorSize), dtype = complex)

    for n in range(0, order_max + 1):
        jn = func(n, kr)
        nnew = nold + 2 * n + 1
        besselfun[:, :, nold:nnew] = jn[..., None]
        nold = nnew
    return besselfun

def get_hankel2_to_order_n(order_max, kr, derivative = False):
    if derivative:
        func = lambda n, kr : dsphankel2(n, kr)
    else:
        func = lambda n, kr : sphankel2(n, kr)
    nold = 0
    NMLocatorSize = (order_max + 1) ** 2
    besselfun = np.zeros((kr.shape[0], kr.shape[1], NMLocatorSize), dtype = complex)

    for n in range(0, order_max + 1):
        jn = func(n, kr)
        nnew = nold + 2 * n + 1
        besselfun[:, :, nold:nnew] = jn[..., None]
        nold = nnew
    return besselfun
# %%
grid_mic_orig = get_eigenmike_grid()
az_mic, elev_mic, r_mic = cart2sph(grid_mic_orig[0],grid_mic_orig[1], grid_mic_orig[2])
# grid_mic_extrap = fib_sphere(5000, radius = 0.3)
grid_mic_extrap = r_mic[0]*get_ico_coords(5).T
grid_ref = _disk_grid_fibonacci(2000, 0.6)

point_source_loc = np.array([2, -2.5, 0.0001])[...,None]

azi_ref, elev_ref, r_ref = cart2sph(grid_ref[0], grid_ref[1], grid_ref[2])

Nmax = 4
NFFT = 127
fs = 48e3
c = 343.
rho = 1.2
delay = 0 # sec
freqs = np.fft.rfftfreq(2*NFFT, d = 1/fs)
omega = 2*np.pi*freqs
k = 2*np.pi*freqs/c
ka = k * r_mic.mean(0)[None, None]  # scatter radius
kr = k * r_ref[..., None]  # scatter radius

if np.any(kr[:, 0] == 0):
    kr[:, 0] = kr[:, 1]
if np.any(ka[:, 0] == 0):
    ka[:, 0] = ka[:, 1]
if np.any(omega[0] == 0):
    omega[0] = omega[1]

Pnm = point_source_spherical_coefficients(Nmax,
                                          k,
                                          point_source_loc,
                                          grid_mic_orig,
                                          sphere_type= 'rigid')


pressure_grid = ispatialFT(Pnm, grid_mic_orig)
pressure_extrap = extrapolate(Pnm, grid_mic_extrap,grid_mic_orig , k)
# pressure_extrap = extrapolate(Pnm, grid_ref, grid_mic, k)


# Pnm_resampled = spatialFT(
#     pressure_grid, grid_ref2, order_max=Nmax
# )


jn_prime_ka = -get_bessel_to_order_n(Nmax, ka, derivative= True) # typo in EFrens paper
hn2_prime_ka = get_hankel2_to_order_n(Nmax, ka, derivative= True)

jn_kr = get_bessel_to_order_n(Nmax, kr, derivative= False)
hn2_kr = get_hankel2_to_order_n(Nmax, kr, derivative= False)

djn_kr = get_bessel_to_order_n(Nmax, kr, derivative= True)
dhn2_kr = get_hankel2_to_order_n(Nmax, kr, derivative= True)

Cm = 1j* np.einsum('rf, rfn, nf -> nf', ka**2, jn_prime_ka, Pnm)
Bm = 1j* np.einsum('rf, rfn, nf -> nf', ka**2, hn2_prime_ka, Pnm)

spherical_harmonic_bases = sph_harm_all(
    Nmax, azi_ref, elev_ref, kind="complex")

p_inc = np.einsum('rfn, nf, rn -> rf', jn_kr, Bm, spherical_harmonic_bases)
p_scat = np.einsum('rfn, nf, rn -> rf', hn2_kr, Cm, spherical_harmonic_bases)

u_inc = -1/(1j * omega * rho) * np.einsum('rfn, nf, rn -> rf', djn_kr, Bm, spherical_harmonic_bases)
u_scat = -1/(1j * omega * rho) * np.einsum('rfn, nf, rn -> rf', dhn2_kr, Cm, spherical_harmonic_bases)
# np.einsum('rfn, nf, rn -> rf', radial_functions, spherical_coefficients, spherical_harmonic_bases)
# %%
I_inc = 0.5 * p_inc * np.conj(u_inc)
I_scat = 0.5 * p_scat * np.conj(u_scat)

r_indx = np.argwhere(r_ref < r_mic.mean(0))

p_inc[r_indx] = 0.
p_scat[r_indx] = 0.
u_inc[r_indx] = 0.
u_scat[r_indx] = 0.
I_inc[r_indx] = 0.
I_scat[r_indx] = 0.
# %% Intensity
fr_ind = 20
costheta = np.cos(azi_ref)
sintheta = np.sin(azi_ref)
Ix_inc = I_inc*costheta[..., None]
Iy_inc = I_inc*sintheta[..., None]

Ix_scat = I_scat*costheta[..., None]
Iy_scat = I_scat*sintheta[..., None]

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.quiver(grid_ref[0], grid_ref[1], Ix_inc[:, fr_ind].real, Iy_inc[:, fr_ind].real)
ax1.set_aspect('equal')

ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

ax2.quiver(grid_ref[0], grid_ref[1], Ix_scat[:, fr_ind].real, Iy_scat[:, fr_ind].real)
ax2.set_aspect('equal')

ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
fig.tight_layout()
fig.show()
# %% Particle velocity
fr_ind = 20
costheta = np.cos(azi_ref)
sintheta = np.sin(azi_ref)
ux_inc = u_inc*costheta[..., None]
uy_inc = u_inc*sintheta[..., None]

ux_scat = u_scat*costheta[..., None]
uy_scat = u_scat*sintheta[..., None]

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.quiver(grid_ref[0], grid_ref[1], ux_inc[:, fr_ind], uy_inc[:, fr_ind])
ax1.set_aspect('equal')

ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

ax2.quiver(grid_ref[0], grid_ref[1], ux_scat[:, fr_ind], uy_scat[:, fr_ind])
ax2.set_aspect('equal')

ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
fig.tight_layout()
fig.show()
# %% plot pressure field (cross sectional)
fr_ind = 20
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1, im = plot_sf(p_inc[:, fr_ind] , grid_ref[0], grid_ref[1], f = freqs[fr_ind],
                 name = r'Incident $k\alpha = {:.2f}$'.format(ka[0, fr_ind]), ax = ax1, colorbar= True, clim = (-0.4, 0.4))
ax2, im = plot_sf(p_scat[:, fr_ind]  , grid_ref[0], grid_ref[1], f = freqs[fr_ind],
                  name=r'Scattered $k\alpha = {:.2f}$'.format(ka[0, fr_ind]), ax = ax2, colorbar= True, clim = (-0.4, 0.4))
circle1 = plt.Circle((0, 0), r_mic.mean(0), color='white')
circle2 = plt.Circle((0, 0), r_mic.mean(0), color='white')
ax1.add_patch(circle1)
ax1.scatter(point_source_loc[0], point_source_loc[1], color = 'k', marker = 'x')
ax2.add_patch(circle2)
ax2.scatter(point_source_loc[0], point_source_loc[1], color = 'k', marker = 'x')
# ax.set_box_aspect((4, 4, 1))
# limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
# ax.set_box_aspect(np.ptp(limits, axis = 1))

fig.tight_layout()
fig.show()
# fig.savefig("./Figures/SoundFieldSeperation_sphere.png", dpi = 300, bbox_inches='tight')
# %% plot particle velocity field (cross sectional)
fr_ind = 20
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1, im = plot_sf(u_inc[:, fr_ind].imag , grid_ref[0], grid_ref[1], f = freqs[fr_ind],
                 name = r'Incident $k\alpha = {:.2f}$'.format(ka[0, fr_ind]), ax = ax1, colorbar= True, clim = (-0.4, 0.4))
ax2, im = plot_sf(u_scat[:, fr_ind].imag  , grid_ref[0], grid_ref[1], f = freqs[fr_ind],
                  name=r'Scattered $k\alpha = {:.2f}$'.format(ka[0, fr_ind]), ax = ax2, colorbar= True, clim = (-0.4, 0.4))
circle1 = plt.Circle((0, 0), r_mic.mean(0), color='white')
circle2 = plt.Circle((0, 0), r_mic.mean(0), color='white')
ax1.add_patch(circle1)
ax1.scatter(point_source_loc[0], point_source_loc[1], color = 'k', marker = 'x')
ax2.add_patch(circle2)
ax2.scatter(point_source_loc[0], point_source_loc[1], color = 'k', marker = 'x')
# ax.set_box_aspect((4, 4, 1))
# limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
# ax.set_box_aspect(np.ptp(limits, axis = 1))

fig.tight_layout()
fig.show()
# fig.savefig("./Figures/SoundFieldSeperation_sphere.png", dpi = 300, bbox_inches='tight')
# %%
from code_SH.SphHarmUtils import set_aspect_equal
fr_ind = 20

fig = plt.figure( figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax, im = plot_array_pressure(pressure_grid[:, fr_ind].real, grid_mic_orig, ax = ax ,z_label= True)
# ax.scatter(point_source_loc[0], point_source_loc[1], point_source_loc[2], color = 'k', marker = 'x')

set_aspect_equal(ax)
ax.view_init(30, 30)

ax = fig.add_subplot(1, 2, 2, projection='3d')

ax, im = plot_array_pressure(pressure_extrap[:, fr_ind].real, grid_mic_extrap, ax = ax ,z_label= True)
# ax.scatter(point_source_loc[0], point_source_loc[1], point_source_loc[2], color = 'k', marker = 'x')
set_aspect_equal(ax)
ax.view_init(30, 30)


fig.tight_layout()
fig.show()


#%%
