import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_stiffness_matrix, assemble_mass_matrix, assemble_matrix, assemble_rhs
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pandas as pd

def main():
    params = ParameterHandler("test_params/convergence_rate_two_zone.toml")

    # h, error, and p arrays for convergence rate calculation
    hs = []
    errors = []
    ps = []

    for _ in range(4):
        mesh = Mesh(params)
        mu = 1.0
        
        hs.append(np.max(mesh.h))

        # assemble mass matrix
        M = assemble_matrix(mu, mesh)
        Q = assemble_rhs(mu, mesh)

        U = sp.linalg.spsolve(M, Q)
        

        exact_fn = lambda x : np.exp(-x) if 0 <= x < 1/2 else np.exp(1/2) * np.exp(-2 * x)

        U_exact = [exact_fn(mesh.vertices[i]) for i in range(mesh.n_vertices)]    
        
        errors.append(np.sum(np.abs(U_exact - U) * mesh.lumped_mass))
        print(f"h: {mesh.h[0]}, l1_error_estimate: {errors[-1]}")

        p = np.log(errors[-1] / errors[-2]) / np.log(hs[-1] / hs[-2]) if len(errors) > 1 else None
        ps.append(p)
        print(f"Estimated convergence rate p: {p}")
              
        params.cells_per_zone *= 2

    convergence_table = pd.DataFrame({
        'h': hs,
        'L1 Error Estimate': errors,
        'Empirical Convergence Rate p': ps
    })

    convergence_table.to_csv('convergence_rate_two_zone.csv', index=False)

    plt.loglog(1 / np.array(hs), errors)

    # plt.plot(mesh.vertices, U)
    # plt.plot(mesh.vertices, U_exact)
    plt.savefig("convergence_rate_two_zone.png")

    print("Test passed!")

if __name__ == "__main__":
    main()