import numpy as np


def rot_m(t, p):
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


def mk3333(M):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == 0 and j == 0:
                        a = 0
                    if i == 1 and j == 1:
                        a = 1
                    if i == 2 and j == 2:
                        a = 2
                    if (i == 1 and j == 2) or (i == 2 and j == 1):
                        a = 3
                    if (i == 0 and j == 2) or (i == 2 and j == 0):
                        a = 4
                    if (i == 0 and j == 1) or (i == 1 and j == 0):
                        a = 5
                    if k == 0 and l == 0:
                        b = 0
                    if k == 1 and l == 1:
                        b = 1
                    if k == 2 and l == 2:
                        b = 2
                    if (k == 1 and l == 2) or (k == 2 and l == 1):
                        b = 3
                    if (k == 0 and l == 2) or (k == 2 and l == 0):
                        b = 4
                    if (k == 0 and l == 1) or (k == 1 and l == 0):
                        b = 5
                    C[i, j, k, l] = M[a, b]
    return C


def mk66(C):
    M = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            if a == 0:
                i = 0
                j = 0
            if a == 1:
                i = 1
                j = 1
            if a == 2:
                i = 2
                j = 2
            if a == 3:
                i = 1
                j = 2
            if a == 4:
                i = 0
                j = 2
            if a == 5:
                i = 0
                j = 1
            if b == 0:
                k = 0
                l = 0
            if b == 1:
                k = 1
                l = 1
            if b == 2:
                k = 2
                l = 2
            if b == 3:
                k = 1
                l = 2
            if b == 4:
                k = 0
                l = 2
            if b == 5:
                k = 0
                l = 1
            M[a, b] = C[i, j, k, l]
    return M


def Rot3333(C, R):
    RC = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    t = 0
                    for p in range(3):
                        for q in range(3):
                            for m in range(3):
                                for n in range(3):
                                    t += R[i, p] * R[j, q] * R[k, m] * R[l, n] * C[p, q, m, n]
                    RC[i, j, k, l] = t
    return RC


def drained_r(psi, theta, s1, s2, s3, p, r0, crit):
    Np = psi.shape[0]
    Nt = theta.shape[0]

    r = np.zeros((Np, Nt))
    pf = p * np.zeros((Np, Nt))
    o = np.ones((Np, Nt))

    sint = np.zeros((Np, Nt))
    sinp = np.zeros((Np, Nt))
    cost = np.zeros((Np, Nt))
    cosp = np.zeros((Np, Nt))

    for i in range(Np):
        for j in range(Nt):
            sint[i, j] = np.sin(theta[j])
            cost[i, j] = np.cos(theta[j])
            sinp[i, j] = np.sin(psi[i])
            cosp[i, j] = np.cos(psi[i])

    sn = s1 * np.square(sinp) * np.square(cost) + s2 * np.square(sinp) * np.square(sint) + s3 * np.square(cosp)

    r = r0 * (o - (sn - p) / crit)

    r[r < 0] = 0.0

    return r


def mkiso(lam, mu):
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


def Hudson81(lam, mu, eps, r, kf):
    #############################################################
    #
    # Function for classic Hudson 1981 fracture model for fluid
    # filled cracks
    #
    # lam     : uncracked lame modulus
    # mu      : uncracked shear modulus
    # eps     : crack density
    # r       : crack aspect ratio
    # kf      : fluid bulk modulus
    #
    # Returns 6X6 stiffness matrix C
    #
    # Written by Mark Chapman, April 2023
    #
    #############################################################

    # create a 6X6 matrix for output

    C = np.zeros((6, 6))

    # create the isotropic reference matrix

    C[0, 0] = lam + 2.0 * mu

    C[0, 1] = lam

    C[0, 2] = lam

    C[1, 0] = lam

    C[1, 1] = lam + 2.0 * mu

    C[1, 2] = lam

    C[2, 0] = lam

    C[2, 1] = lam

    C[2, 2] = lam + 2.0 * mu

    C[3, 3] = mu

    C[4, 4] = mu

    C[5, 5] = mu

    # work out the fluid term

    t1 = kf * (lam + 2.0 * mu)

    t2 = np.pi * r * mu * (lam + mu)

    Kc = t1 / t2

    # work out the crack compliances

    U1 = (16.0 / 3.0) * (lam + 2.0 * mu) / (3.0 * lam + 4.0 * mu)

    t1 = 4.0 * (lam + 2.0 * mu)

    t2 = 3.0 * (lam + mu) * (1.0 + Kc)

    U3 = t1 / t2

    # calculate the correction terms

    c11 = -1.0 * np.power(lam, 2) * eps * U3 / mu

    c13 = -1.0 * lam * (lam + 2.0 * mu) * eps * U3 / mu

    c33 = -1.0 * np.power((lam + 2.0 * mu), 2) * eps * U3 / mu

    c44 = -1.0 * mu * eps * U1

    # correct the matrix C

    C[0, 0] = C[0, 0] + c11

    C[1, 1] = C[1, 1] + c11

    C[2, 2] = C[2, 2] + c33

    C[0, 2] = C[0, 2] + c13

    C[1, 2] = C[1, 2] + c13

    C[2, 0] = C[2, 0] + c13

    C[2, 1] = C[2, 1] + c13

    C[3, 3] = C[3, 3] + c44

    C[4, 4] = C[4, 4] + c44

    # now need to get C[0, 1] and C[1, 0] from the corrected components

    C[0, 1] = C[0, 0] - 2 * C[5, 5]

    C[1, 0] = C[0, 1]

    # finish

    return C


def pm6(M):
    for i in range(6):
        row = " ".join([f"{M[i, j]:4.2e}" for j in range(6)])
        print(f"[ {row} ]")












