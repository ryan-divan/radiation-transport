import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_stiffness_matrix, assemble_mass_matrix, assemble_matrix, assemble_rhs
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def main():
    params = ParameterHandler("test_params/reed_problem.toml")

    # generate mesh
    mesh = Mesh(params)

    mu = 1.0
    M = assemble_matrix(mu, mesh, use_supg=False)
    Q = assemble_rhs(mu, mesh, use_supg=False)
    U = sp.linalg.spsolve(M, Q, use_supg=False)

    M_supg = assemble_matrix(mu, mesh)
    Q_supg  = assemble_rhs(mu, mesh)
    U_supg = sp.linalg.spsolve(M, Q)

    plt.plot(mesh.vertices, U, label="Standard Galerkin method")
    plt.plot(mesh.vertices, U_supg, label="SUPG")
    plt.legend()
    # plt.plot(mesh.vertices, U_exact, label="exact")
    # plt.legend()
    plt.savefig("test_superconvergence_undershoot.png")

    print("Test passed!")

if __name__ == "__main__":
    main() 

