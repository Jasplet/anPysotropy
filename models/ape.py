import numpy as np




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
