import numpy as np

from apepy import ape as ap


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
    H_base = ap.Hudson81(material['lam'],
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


if __name__ == '__main__':

    # define the input applied stress parameters and pore-fluid pressure
    # stress in the x1, x2 and x3 axes. x3 is vertical
    s3 = 500000000
    s2 = 100000000
    s1 = 10000
    p = 3000
    # define the rock, crack and fluid paramters
    material_params = {
        'lam': 2e10,  # lame parameter
        'mu': 2e10,   # lame parameter
        'kf': 2e9,    # fluid builk modulus
        'r0': 0.001,  # initial crack aspect ratio
        'eps': 0.05   # crack density
    }

    # calculate dependent parameters
    v = material_params['lam'] / (2.0*(material_params['lam'] + material_params['mu']))
    crit = np.pi*material_params['mu']*material_params['r0']/(2.0*(1.0 - v))
    material_params['crit'] = crit
    material_params['v'] = v

    # calculate the anisotropic APE tensor
    M = drained_ape(s1,
                    s2,
                    s3,
                    p,
                    material_params)
    np.to_npz('example_drained_ape_tensor.npz', M)
    ap.pm6(M)
