import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.discrete_ordinates import assemble_sigma_matrix
import numpy as np
import scipy.sparse as sp

def main():
    params = ParameterHandler("test_params/simple_source_iteration.toml")

    # generate mesh
    mesh = Mesh(params)
    mu = 1.0

    # assemble stiffness matrix
    sigma_matrix = assemble_sigma_matrix(mu, mesh)
    A = sp.csr_matrix.todense(sigma_matrix)

    print(A)
    print(4 * np.pi * A)

    # # invertible
    # assert(np.all(np.linalg.eigvals(A) != 0))

    # # tridiagonal
    # assert(np.all(np.triu(A, k=2) == 0))
    # assert(np.all(np.tril(A, k=-2) == 0))

    # # check that the entries are correct
    # assert(np.isclose(A[0,0], - 1/2 * mu))
    # assert(np.isclose(A[0,1], -1/2 * mu))
    # assert(np.isclose(A[1,0], 1/2 * mu))
    # assert(np.isclose(A[1,1], 0))

    # assert(np.isclose(A[1,2], -1/2 * mu))
    # assert(np.isclose(A[2,1], 1/2 * mu))
    # assert(np.isclose(A[2,2], 0))
    # assert(np.isclose(A[2,3], -1/2 * mu))
    # assert(np.isclose(A[3,2], 1/2 * mu))

    # assert(np.isclose(A[3,3], 0))
    # assert(np.isclose(A[3,4], -1/2 * mu))
    # assert(np.isclose(A[4,3], 1/2 * mu))
    # assert(np.isclose(A[4,4], 1/2 * mu))

    print("Test passed!")

if __name__ == "__main__":
    main() 

