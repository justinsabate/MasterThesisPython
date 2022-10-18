"""Pseudocode as a first step towards the minimization problem"""
import numpy as np
import matrix
from code_SH.SphHarmUtils import sph_harm_all, stack_real_imag_Y
N = 4
azi =
elev =


'''Formulation 1'''

'''Calculate all the needed matrices'''
'1 spherical harmonic basis up to order Na'

'2 High frequency time aligned HRTF set'
## 2a Time alignment of the HRTF
## 2b Expansion in the spherical harmonic domain

'3 Real valued frequency independent diagonal weighting matrix of the quadrature weights'
## might need this
# grid = get_eigenmike_grid(plot=False) returns x,y,z (positions)
## or something to do with this
# r = 0.084 / 2
# c = 343
# NFFT = 4096
# fs_min = 48000
# freq = np.arange(0, NFFT // 2 + 1) * (fs_min / NFFT)
# kr = 2 * np.pi * freq * r / c
# weights(N, kr, 'rigid') ## this returns N+1 x len(freq) weighting function (influence of the rigid sphere)

'4 Calculate RH'
## calculate RH = H*W*np.matrix.H(H)

'5 Calculate RY'
## spatial covariance matrix

'6 Calculate Q'
## such as QHQ=I

'7 Calculate C'
## find C such as ChC = RH

'''Last formulation'''
'The expression of the binaural renderer is given '

'1 Calculate C'
H =  # OK
W =
RH = H*W*np.matrix.H(H)
C =  # suitable matrix decomposition of RH such that Hermitian(C)*C = RH
H_circle =  # OK
A = np.matrix.H(np.ones((2,2)))
B = matrix.decompose(A,return_type='LL') # returns L such as L*Hermitian(L) = A ?


'2.a Calculate the spherical harmonics basis'
spherical_harmonic_bases = sph_harm_all(
    N, azi, elev, kind='complex'
)
# spherical harmonics basis
YNaP = stack_real_imag_Y(spherical_harmonic_bases)

'2 Calculate A'
A = C*H_circle*W*np.matrix.H(YNaP)

'3 Singular value decomposition of A'
U,Sigma,V = np.linalg.svd(A)

'4 Calculate the binaural renderer BNaTAC'
Lambda = np.transpose(np.zeros((2,(N+1)**2)))
Lambda[0, 0] = 1
Lambda[1, 1] = 1
# BNaTAC = V * Lambda * np.matrix.H(U) * C

