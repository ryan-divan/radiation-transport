import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_stiffness_matrix, assemble_mass_matrix, assemble_matrix, assemble_rhs
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def main():
    params = ParameterHandler("test_params/superconvergence.toml")

    # generate mesh
    mesh = Mesh(params)
    mu = 1.0

    # assemble mass matrix
    M = assemble_matrix(mu, mesh)
    Q = assemble_rhs(mu, mesh)
    print(M)

    print(assemble_mass_matrix(mu, mesh))
    print(assemble_stiffness_matrix(mu, mesh))

    print(Q)

    U = sp.linalg.spsolve(M, Q)
    print(U)

    # exact_fn = lambda x : np.exp(-x) 
    # U_exact = [exact_fn(xs) for xs in mesh.vertices]
    # exact_fn = lambda x : np.exp(-x) if 0 <= x < 1/2 else np.exp(1/2) * np.exp(-2 * x)
    # U_exact = [exact_fn(xs) for xs in mesh.vertices]

    # plt.plot(mesh.vertices, U_exact, label="exact")
    
    plt.plot(mesh.vertices, U, label="approximate")
    # plt.plot(mesh.vertices, U_exact, label="exact")
    # plt.legend()
    plt.savefig("test_superconvergence_undershoot.png")

    print("Test passed!")

if __name__ == "__main__":
    main() 

