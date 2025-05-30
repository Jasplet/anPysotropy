import numpy as np
import numba
import itertools


def make_rotation_matrix(t, p):
    '''
    Creates a 3x3 rotation matrix for the angles
    t (theta) and p (psi)
    '''
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(p) * np.cos(t)
    R[0, 1] = -1.0 * np.sin(t)
    R[0, 2] = -1.0 * np.sin(p) * np.cos(t)
    R[1, 0] = np.cos(p) * np.sin(t)
    R[1, 1] = np.cos(t)
    R[1, 2] = -1.0 * np.sin(p) * np.sin(t)
    R[2, 0] = np.sin(p)
    R[2, 1] = 0.0
    R[2, 2] = np.cos(p)
    return R


def voight_index_from_3x3_tensor(i, j):
    '''
    Converts indices of the full 3x3x3x3 elastic tensor
    to Voight incicies
    '''
    if i > 5 or j > 5:
        raise ValueError(f'Invalid index: {i}, {j}. Dont forget python indexes start at 0')
    if i == j:
        idx = i
    else:
        idx = 6 - i - j
    return idx


def voight_6x6_to_elastic_3x3_tensor(C_voight):

    C_voight = np.asarray(C_voight)
    if C_voight.shape != (6, 6):
        raise ValueError("C_voight must be a 6x6 matrix")

    Cijkl = np.zeros((3, 3, 3, 3), dtype=float)

    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        m = voight_index_from_3x3_tensor(i, j)
        n = voight_index_from_3x3_tensor(k, l)
        Cijkl[i, j, k, l] = C_voight[m, n]

    return Cijkl


def elastic_3x3_tensor_to_voight(Cijkl):

    Cijkl = np.asarray(Cijkl)
    if Cijkl.shape != (3, 3, 3, 3):
        raise ValueError("Cijkl must be a 3x3x3x3 tensor")

    C_voight = np.zeros((6, 6))
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        m = voight_index_from_3x3_tensor(i, j)
        n = voight_index_from_3x3_tensor(k, l)
        C_voight[m, n] = Cijkl[i, j, k, l]
    return C_voight

def pretty_print_C_voight(M):
    for i in range(6):
        row = " ".join([f"{M[i, j]:4.2e}" for j in range(6)])
        print(f"[ {row} ]")

### The following functions have been ported from the Matlab Seismic Anisotropy Toolkit (MSAT)
# Copyright (c) 2011, James Wookey and Andrew Walker
# All rights reserved.
# 
# Redistribution and use in source and binary forms, 
# with or without modification, are permitted provided 
# that the following conditions are met:
# 
#    * Redistributions of source code must retain the 
#      above copyright notice, this list of conditions 
#      and the following disclaimer.
#    * Redistributions in binary form must reproduce 
#      the above copyright notice, this list of conditions 
#      and the following disclaimer in the documentation 
#      and/or other materials provided with the distribution.
#    * Neither the name of the University of Bristol nor the names 
#      of its contributors may be used to endorse or promote 
#      products derived from this software without specific 
#      prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS 
# AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@numba.jit(nopython=True)
def rotate_tensor(Cijkl, R):
    '''
    Rotates 4th order 3x3x3x3 tensor T by a 3x3 
    rotation matrix rotmat.

    Using numba significantly speeds up the process
    of rotating the tensor.
    '''
    rot_Cijkl = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    t = 0
                    for p in range(3):
                        for q in range(3):
                            for m in range(3):
                                for n in range(3):
                                    t += R[i, p] * R[j, q] \
                                         * R[k, m] * R[l, n] \
                                         * Cijkl[p, q, m, n]
                    rot_Cijkl[i, j, k, l] = t
    return rot_Cijkl

def make_iso_Cijkl(lam, mu):
    '''
    Creates a 3x3x3x3 isotropic elastic tensor
    from the Lame parameters lam and mu.
    '''
    C = np.zeros((3, 3, 3, 3))
    C[0, 0, 0, 0] = lam + 2.0 * mu
    C[1, 1, 1, 1] = lam + 2.0 * mu
    C[2, 2, 2, 2] = lam + 2.0 * mu
    C[0, 0, 1, 1] = lam
    C[0, 0, 2, 2] = lam
    C[1, 1, 0, 0] = lam
    C[1, 1, 2, 2] = lam
    C[2, 2, 0, 0] = lam
    C[2, 2, 1, 1] = lam
    C[1, 2, 1, 2] = mu
    C[1, 2, 2, 1] = mu
    C[2, 1, 1, 2] = mu
    C[2, 1, 2, 1] = mu
    C[0, 2, 0, 2] = mu
    C[0, 2, 2, 0] = mu
    C[2, 0, 0, 2] = mu
    C[2, 0, 2, 0] = mu
    C[0, 1, 0, 1] = mu
    C[0, 1, 1, 0] = mu
    C[1, 0, 0, 1] = mu
    C[1, 0, 1, 0] = mu
    return C

def decompose_C_voight(C_voight):
    '''
    Decomposes a 6x6 Voight matrix into its isotropic and anisotropic parts.
    Following Browaeys and Chevrot (2004).

    This function is ported from the function MS_decomp from the 
    Matlab Seismic Anisotropy Toolkit (MSAT) by J Wookey and A Walker.

    References:
     Browaeys, J. T. and S. Chevrot (2004) Decomposition of the elastic
        tensor and geophysical applications. Geophysical Journal 
        international v159, 667-678.

    '''
    C_voight = np.asarray(C_voight)
    if C_voight.shape != (6, 6):
        raise ValueError("C_voight must be a 6x6 matrix")

    elastic_vectorX = make_elastic_vectorX(C_voight)
    # Isotropic part
    lam = (C_voight[0, 0] + C_voight[1, 1] + C_voight[2, 2]) / 3.0
    mu = (C_voight[3, 3] + C_voight[4, 4] + C_voight[5, 5]) / 3.0
    C_iso = make_iso_Cijkl(lam, mu)

    # Anisotropic part
    C_aniso = C_voight - elastic_3x3_tensor_to_voight(C_iso)

    return C_iso, C_aniso


def C_voight_to_elastic_vector(C_voight):
    '''
    Following equation 2.2 of Browaeys and Chevrot (2004)
    Decomposes a 6x6 Voight matrix into a 21 element elastic vector

    Note that as Python uses 0-based indexing, the indices
    in the vector X are shifted by 1 compared to the
    indices in the original paper and the Matlab code.

    This is a port of the function C2X from the MSAT. 
    '''

    X = np.zeros(21)
    C = C_voight
    X[0]  = C[0, 0]
    X[1]  = C[1, 1]
    X[2]  = C[2, 2]
    X[3]  = np.sqrt(2) * C[1, 2]
    X[4]  = np.sqrt(2) * C[0, 2]
    X[5]  = np.sqrt(2) * C[0, 1]
    X[6]  = 2.0 * C[3, 3]
    X[7]  = 2.0 * C[4, 4]
    X[8]  = 2.0 * C[5, 5]
    X[9]  = 2.0 * C[0, 3]
    X[10] = 2.0 * C[1, 4]
    X[11] = 2.0 * C[2, 5]
    X[12] = 2.0 * C[2, 3]
    X[13] = 2.0 * C[0, 4]
    X[14] = 2.0 * C[1, 5]
    X[15] = 2.0 * C[1, 3]
    X[16] = 2.0 * C[2, 4]
    X[17] = 2.0 * C[0, 5]
    X[18] = 2.0 * np.sqrt(2) * C[4, 5]
    X[19] = 2.0 * np.sqrt(2) * C[3, 5]
    X[20] = 2.0 * np.sqrt(2) * C[3, 4]

    return X

def elastic_vector_to_C_voight(X):
    '''
    Converts a 21 element elastic vector to a 6x6 Voight matrix.
    This is the inverse of C_voight_to_elastic_vector.

    Note that as Python uses 0-based indexing, the indices
    in the vector X are shifted by 1 compared to the
    indices in the original paper and the Matlab code.

    This is a port of the function X2C from the MSAT. 
    '''

    C = np.zeros((6, 6))
    C[0, 0] = X[0]
    C[1, 1] = X[1]
    C[2, 2] = X[2]
    C[1, 2] = X[3] / np.sqrt(2)
    C[0, 2] = X[4] / np.sqrt(2)
    C[0, 1] = X[5] / np.sqrt(2)
    C[3, 3] = X[6] / 2.0
    C[4, 4] = X[7] / 2.0
    C[5, 5] = X[8] / 2.0
    C[0, 3] = X[9] / 2.0
    C[1, 4] = X[10] / 2.0
    C[2, 5] = X[11] / 2.0
    C[2, 3] = X[12] / 2.0
    C[0, 4] = X[13] / 2.0
    C[1, 5] = X[14] / 2.0
    C[1, 3] = X[15] / 2.0
    C[2, 4] = X[16] / 2.0
    C[0, 5] = X[17] / 2.0
    C[4, 5] = X[18] / (2.0 * np.sqrt(2))
    C[3, 5] = X[19] / (2.0 * np.sqrt(2))
    C[3, 4] = X[20] / (2.0 * np.sqrt(2))

    # Symmetrize the matrix
    for i in range(6):
        for j in range(i + 1, 6):
            C[j, i] = C[i, j]
    # Check if the matrix is symmetric
    if not np.allclose(C, C.T):
        raise ValueError("The resulting matrix is not symmetric")
    return C








