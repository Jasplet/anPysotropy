from anPysotropy.utils import matrix_tools

import numpy as np
import pytest
import itertools
from utils.matrix_tools import voight_6x6_to_elastic_3x3_tensor

def test_output_shape():
    C = np.eye(6)
    tensor = voight_6x6_to_elastic_3x3_tensor(C)
    assert tensor.shape == (3, 3, 3, 3)


def test_diagonal_mapping():
    C = np.eye(6)
    tensor = voight_6x6_to_elastic_3x3_tensor(C)
    # Check a few diagonal elements
    assert tensor[0,0,0,0] == 1
    assert tensor[1,1,1,1] == 1
    assert tensor[2,2,2,2] == 1


def test_invalid_shape():
    C = np.eye(5)
    with pytest.raises(ValueError):
        voight_6x6_to_elastic_3x3_tensor(C)


def test_single_element_mapping():
    C = np.zeros((6,6))
    C[2,3] = 42
    tensor = voight_6x6_to_elastic_3x3_tensor(C)
    # Find the corresponding indices for (2,3)
    found = False
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if tensor[i,j,k,l] == 42:
                        found = True
    assert found


def test_full_mapping():
    C = np.random.randint(1, 20, size=(6, 6))
    tensor = voight_6x6_to_elastic_3x3_tensor(C)
    # Check if all elements are mapped correctly
    for i, j, k ,l in itertools.product(range(3), range(3), range(3), range(3)):
        m = matrix_tools.voight_index_from_3x3_tensor(i, j)
        n = matrix_tools.voight_index_from_3x3_tensor(k, l)
        assert tensor[i, j, k, l] == C[m, n]


def test_rotation_matrix():
    t = np.pi / 4
    p = np.pi / 4
    R = matrix_tools.make_rotation_matrix(t, p)
    # Check the shape of the rotation matrix
    assert R.shape == (3, 3)
    # Check if the rotation matrix is orthogonal
    assert np.allclose(np.dot(R, R.T), np.eye(3))
    expected_R = np.array([[np.cos(p) * np.cos(t), -np.sin(t), -np.sin(p) * np.cos(t)],
                           [np.cos(p) * np.sin(t), np.cos(t), -np.sin(p) * np.sin(t)],
                           [np.sin(p), 0, np.cos(p)]])
    assert np.allclose(R, expected_R)

