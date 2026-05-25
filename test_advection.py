import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from scipy.sparse.linalg import spsolve
from source.advection import assemble_matrix, assemble_rhs


def main():
    # check that the parameters file is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python3 test_mesh.py <parameter_file>")
        sys.exit(1)

    # load parameters
    params = ParameterHandler(sys.argv[1])

    # generate mesh
    mesh = Mesh(params)
    mu = 1
    # T = assemble_matrix(mu, mesh)
    print(params)
    print(mesh)
    # Q = assemble_rhs(mu, mesh)
    M = assemble_matrix(mu, mesh)

    print(M)
    # U = spsolve(T, Q)
    # print(U)

    # print parameters and mesh information


if __name__ == "__main__":
    main()
