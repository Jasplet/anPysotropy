import numpy as np

from apepy import ape as ap


def undrained_ape(s1,
                  s2,
                  s3,
                  p,
                  material,
                  npsi=100,
                  ntheta=200,
                  npressures=1000):
    '''
    Calculates an anisotropic APE tensor (6x6 Voight notiation form)
    for undrained cracks in a rock matrix. Hudson (1981)'s model is
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

    # do a scan for the induced pressure
    ps = np.linspace(0,
                   np.log10(material['mu']),
                   npressures)

    obj = np.zeros((npressures,))

    for m in range(npressures):
        pt = np.power(10, ps[m])
        g = ap.drained_r(psi,
                         theta,
                         s1,
                         s2,
                         s3,
                         pt,
                         material['r0'],
                         material['crit'])
        g = g/material['r0']
        # vg = 0.0
        # for i, ipsi in enumerate(npsi):
        #     for j in range(ntheta):
        #         vg = vg + g[i, j]*dt*dp*np.sin(ipsi)/(4.0*np.pi)
        # vecotrised version of the above
        # np.newaxis is used to make psi a column vector to make sure 
        # each row of g is multiplied by the corresponding element of psi        
        vg = np.sum(g * dt * dp * np.sin(psi[:,np.newaxis]) / (4.0*np.pi))
        vf = np.exp(-1.0 * pt / material['kf'])
        obj[m] = np.abs(vg-vf)

    # find the equilibrium pressure by minimising the objective

    idxmin = np.argmin(obj)
    p = np.power(10, ps[idxmin])

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
        for j in range(0,):
            pp = psi[i]
            tt = theta[j]
            r = g[i,j]
            if np.isclose(r, 0):
               pass
            else:
                R = ap.rot_m(-1.0*tt, -1.0*pp)
                r = r * r0
                H = H_base.copy()
                dH = (H - M0)*dt*dp*np.sin(pp)/(4.0*np.pi)
                dH4 = ap.mk3333(dH)
                dH4r = ap.Rot3333(dH4, R)
                C0 = C0 + dH4r
    # convert back to 6x6 Voight notation matrix for output
    M = ap.mk66(C0)
    # print the matrix
    ap.pm6(M)
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
    crit = np.pi*material_params['mu']*material_params['r0']/(2.0*(1.0 -v))
    material_params['crit'] = crit
    material_params['v'] = v




