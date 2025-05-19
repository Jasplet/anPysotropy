import numpy as np
from anPysotropy.models.hudson import make_hudson_tensor
from anPysotropy.utils.matrix_tools import make_iso_Cijkl, rotate_tensor
from anPysotropy.utils.matrix_tools import elastic_3x3_tensor_to_voight, make_rotation_matrix, voight_6x6_to_elastic_3x3_tensor


def drained_r(psi,
              theta,
              s1,
              s2,
              s3,
              p,
              r0,
              crit):
    '''
    Calculates aspect ratio distrubution for drained case
    in APE model (Zatespin et al., 1997)
    '''
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
    # equation 5.38  in Zatespin and Crampin (1997).
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
    # make the background isotropic tensor
    C_ijkl = make_iso_Cijkl(material['lam'],
                            material['mu'])
    # convert of 6x6 Voight notation
    C_iso_voight = elastic_3x3_tensor_to_voight(C_ijkl)

    psis = np.linspace(0, np.pi, npsi)
    thetas = np.linspace(0, 2.0*np.pi, ntheta)
    dpsi = psis[1]-psis[0]
    dtheta = thetas[1]-thetas[0]

    # calculate the aspect ratio distribution
    g = drained_r(psis,
                  thetas,
                  s1,
                  s2,
                  s3,
                  p,
                  material['r0'],
                  material['crit'])
    # calculate the anisotropic elastic tensor
    # calc base hudson tensor outside of loops
    # is it does not depend on anything inside.
    H_base = make_hudson_tensor(material['lam'],
                                material['mu'],
                                material['cden'],
                                material['aspect_ratio'],
                                material['kf'],
                                mup=0)
    # lam     : uncracked lame modulus
    # mu      : uncracked shear modulus
    # eps     : crack density
    # r       : crack aspect ratio
    # kf      : fluid bulk modulus
    for i in range(0, npsi):
        for j in range(0, ntheta):
            psi = psis[i]
            theta = thetas[j]
            r = g[i, j]
            if np.isclose(r, 0):
                pass
            else:
                R = make_rotation_matrix(-1.0*theta, -1.0*psi)
                r = r * material['r0']
                # copy H_base so we can modify it without changing the original
                H = H_base.copy()
                dH = (H - C_iso_voight) * dtheta * dpsi * np.sin(psi) / (4.0 * np.pi)
                dH4 = voight_6x6_to_elastic_3x3_tensor(dH)
                dH4r = rotate_tensor(dH4, R)
                C_ijkl = C_ijkl + dH4r

    return elastic_3x3_tensor_to_voight(C_ijkl)


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
    psis = np.linspace(0, np.pi, npsi)
    thetas = np.linspace(0, 2.0*np.pi, ntheta)

    dpsi = psis[1] - psis[0]
    dtheta = thetas[1] - thetas[0]

    # make the background isotropic tensor
    C_ijkl = make_iso_Cijkl(material['lam'],
                            material['mu'])
    # convert of 6x6 Voight notation
    C_iso_voight = elastic_3x3_tensor_to_voight(C_ijkl)

    # do a scan for the induced pressure
    pressures = np.linspace(0,
                            np.log10(material['mu']),
                            npressures)

    obj = np.zeros((npressures,))

    for m in range(npressures):
        pt = np.power(10, pressures[m])
        g = drained_r(psis,
                      thetas
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
        vg = np.sum(g * dtheta * dpsi * np.sin(psis[:, np.newaxis]) / (4.0*np.pi))
        vf = np.exp(-1.0 * pt / material['kf'])
        obj[m] = np.abs(vg-vf)

    # find the equilibrium pressure by minimising the objective

    idxmin = np.argmin(obj)
    p = np.power(10, pressures[idxmin])

    # calculate the aspect ratio distribution
    g = drained_r(psis,
                  thetas,
                  s1,
                  s2,
                  s3,
                  p,
                  material['r0'],
                  material['crit'])
    # calculate the anisotropic elastic tensor
    # calc base hudson tensor outside of loops
    # is it does not depend on anything inside.
    H_base = make_hudson_tensor(material['lam'],
                                material['mu'],
                                material['cden'],
                                material['aspect_ratio'],
                                material['kf'],
                                mup=0)
    for i in range(0, npsi):
        for j in range(0,):
            psi = psis[i]
            theta = thetas[j]
            r = g[i, j]
            if np.isclose(r, 0):
                pass
            else:
                R = make_rotation_matrix(-1.0*theta, -1.0*psi)
                r = r * material['r0']
                H = H_base.copy()
                dH = (H - C_iso_voight)*dtheta*dpsi*np.sin(psi)/(4.0*np.pi)
                dH4 = voight_6x6_to_elastic_3x3_tensor(dH)
                dH4r = rotate_tensor(dH4, R)
                C0 = C0 + dH4r
    # convert back to 6x6 Voight notation matrix for output

    return elastic_3x3_tensor_to_voight(C0)
