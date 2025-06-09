import numpy as np


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
