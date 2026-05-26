import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_mass_matrix
import numpy as np
import scipy.sparse as sp

def main():
    params = ParameterHandler("test_params/very_simple_params.toml")

    # generate mesh
    mesh = Mesh(params)
    mu = None # not used in mass matrix assembly, but required by the function signature

    # assemble mass matrix
    mass_matrix = assemble_mass_matrix(mu, mesh)
    A = sp.csr_matrix.todense(mass_matrix)

    # SPD
    assert(np.allclose(A, A.T))
    assert(np.all(np.linalg.eigvals(A).dtype == np.float64))
    assert(np.all(np.linalg.eigvals(A) > 0))

    # tridiagonal
    assert(np.all(np.triu(A, k=2) == 0))
    assert(np.all(np.tril(A, k=-2) == 0))

    # check that the entries are correct
    assert(np.isclose(A[0,0], 1/6))
    assert(np.isclose(A[0,1], 1/12))
    assert(np.isclose(A[1,0], 1/12))
    assert(np.isclose(A[1,1], 1/3))

    assert(np.isclose(A[1,2], 1/12))
    assert(np.isclose(A[2,1], 1/12))
    assert(np.isclose(A[2,2], 1/2))
    assert(np.isclose(A[2,3], 1/6))
    assert(np.isclose(A[3,2], 1/6))

    assert(np.isclose(A[3,3], 2/3))
    assert(np.isclose(A[3,4], 1/6))
    assert(np.isclose(A[4,3], 1/6))
    assert(np.isclose(A[4,4], 1/3))

    print("Test passed!")

if __name__ == "__main__":
    main() 

