import unittest
import numpy as np
from anPysotropy.models import hudson


def test_hudson_make_c0():
    """
    Test the make_c0 function from the Hudson model.
    """
    # Define the input parameters
    lam = 4
    mu = 1

    # Call the function to test
    c0_a = hudson.make_hudson_c0(lam, mu)

    # Define the expected output
    expected_c0_a = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0],
                             [lam, lam + 2 * mu, lam, 0, 0, 0],
                             [lam, lam, lam + 2 * mu, 0, 0, 0],
                             [0, 0, 0, mu, 0, 0],
                             [0, 0, 0, 0, mu, 0],
                             [0, 0, 0, 0, 0, mu]])
    mu2 = 0
    c0_b = hudson.make_hudson_c0(lam, mu2)
    expected_c0_b = np.array([[lam + 2 * mu2, lam, lam, 0, 0, 0],
                             [lam, lam + 2 * mu2, lam, 0, 0, 0],
                             [lam, lam, lam + 2 * mu2, 0, 0, 0],
                             [0, 0, 0, mu2, 0, 0],
                             [0, 0, 0, 0, mu2, 0],
                             [0, 0, 0, 0, 0, mu2]])

    # Check if the output matches the expected output
    np.testing.assert_array_almost_equal(c0_a, expected_c0_a)
    np.testing.assert_array_almost_equal(c0_b, expected_c0_b)
    # Check if the output is a 6x6 matrix
    assert c0_a.shape == (6, 6), "Output is not a 6x6 matrix"
    # Check if the output is a numpy array
    assert isinstance(c0_a, np.ndarray), "Output is not a numpy array"
    # Check if the output is positive definite
    assert np.all(np.linalg.eigvals(c0_a) > 0), "Output is not positive definite"


def test_make_hudson_tensor_real():
    """
    Test the make_hudson_tensor function from the Hudson model.
    """
    # Define the input parameters
    vp = 5e3
    vs = 3e3
    rho = 2.5e3
    mu = rho*vs**2
    lam = rho*vp**2 - 2*mu
    kappap = 2.2e9
    mup = 0
    cden = 0.1
    aspect = 1e-3
    # Call the function to test
    tensor = hudson.make_hudson_tensor(lam,
                                       mu,
                                       rho,
                                       kappap,
                                       mup,
                                       cden,
                                       aspect,
                                       return_complex=False)

    # Check if the output is a numpy array
    assert isinstance(tensor, np.ndarray), "Output is not a numpy array"
    # Check if the output is a 6x6 matrix
    assert tensor.shape == (6, 6), "Output is not a 6x6 matrix"
    # Check if the output is symmetric
    assert np.allclose(tensor, tensor.T), "Output is not symmetric"
    # test if output is real
    assert np.isrealobj(tensor), "Output is not real"


def test_make_hudson_tensor_complex():
    """
    Test the make_hudson_tensor function from the Hudson model with complex output.
    """
    # Define the input parameters
    vp = 5e3
    vs = 3e3
    rho = 2.5e3
    mu = rho*vs**2
    lam = rho*vp**2 - 2*mu
    kappap = 2.2e9
    mup = 0
    cden = 0.1
    aspect = 1e-3
    # Call the function to test
    tensor = hudson.make_hudson_tensor(lam,
                                       mu,
                                       rho,
                                       kappap,
                                       mup,
                                       cden,
                                       aspect,
                                       crad=10,
                                       freq=1,
                                       return_complex=True)

    # Check if the output is a numpy array
    assert isinstance(tensor, np.ndarray), "Output is not a numpy array"
    # Check if the output is a 6x6 matrix
    assert tensor.shape == (6, 6), "Output is not a 6x6 matrix"
    # Check if the output is symmetric
    assert np.allclose(tensor, tensor.T), "Output is not symmetric"
    # test if output is complex
    assert np.iscomplexobj(tensor), "Output is not complex"