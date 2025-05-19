import numpy as np
from anPysotropy.utils.matrix_tools import mkiso, mk66
from anPysotropy.models.hudson import hudson_c_real


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



def drained_ape(s1,
                s2,
                s3,
                p,
                material,
                npsi=100,
                ntheta=200):
    '''
    Calculates an anisotropic APE tensor (6x6 Voight notiation form) for drained
    cracks in a rock matrix. Hudson (1981)'s model is
    used for crack anisotropy.
    '''
    psi = np.linspace(0, np.pi, npsi)
    theta = np.linspace(0, 2.0*np.pi, ntheta)

    dp = psi[1]-psi[0]

    dt = theta[1]-theta[0]

    # make the background isotropic tensor
    C0 = ap.mkiso(material['lam'],
                  material['mu'])
    # convert of 6x6 Voight notation
    M0 = ap.mk66(C0)

    # calculate the aspect ratio distribution
    g = ap.drained_r(psi,
                     theta,
                     s1,
                     s2,
                     s3,
                     p,
                     material['r0'],
                     material['crit'])
    # calculate the anisotropic elastic tensor
    # calc base hudson tensor outside of loops
    # is it does not depend on anything inside.
    H_base = hudson_c_real(material['lam'],
                           material['mu'],
                           material['eps'],
                           material['r'],
material['kf'])
    for i in range(0, npsi):
        for j in range(0, ntheta):
            pp = psi[i]
            tt = theta[j]
            r = g[i, j]
            if np.isclose(r, 0):
                pass
            else:
                R = ap.rot_m(-1.0*tt, -1.0*pp)
                r = r * material['r0']
                # copy H_base so we can modify it without changing the original
                H = H_base.copy()
                dH = (H - M0)*dt*dp*np.sin(pp)/(4.0*np.pi)
                dH4 = ap.mk3333(dH)
                dH4r = ap.Rot3333(dH4, R)
                C0 = C0 + dH4r

    M = ap.mk66(C0)

    return M
