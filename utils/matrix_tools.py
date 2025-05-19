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
    if i == j:
        idx = i
    else:
        idx = 9 - i - j
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


def pretty_print_C_voight(M):
    for i in range(6):
        row = " ".join([f"{M[i, j]:4.2e}" for j in range(6)])
        print(f"[ {row} ]")












