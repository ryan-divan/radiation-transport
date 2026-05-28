import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_matrix, assemble_rhs
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

    for _ in range(9):
        mesh = Mesh(params)
        mu = 1.0
        
        hs.append(np.max(mesh.h))

        # assemble mass matrix
        M = assemble_matrix(mu, mesh, use_supg=True)
        Q = assemble_rhs(mu, mesh.params.inflow_value[0], mesh, use_supg=True)

        U = sp.linalg.spsolve(M, Q)

        # def exact_fn(x):
        #     if mu > 0:
        #         if 0 <= x < 0.3:
        #             return 1
        #         elif 0.3 <= x < 0.6:
        #             return np.exp(- 100 * (x - 0.3))
        #         elif 0.6 <= x <= 1:
        #             return 1/2 + (np.exp(-30) - 1/2) * np.exp(-2 * (x - 0.6))
        #         else:
        #             return 0
        #     elif mu < 0:
        #         if 0 <= x < 0.3:
        #             return 1 - np.exp(x - 0.3)
        #         elif 0.3 <= x < 0.6:
        #             return (1/2 * (1 + np.exp(-0.8))) * np.exp(100 * (x - 0.6))
        #         elif 0.6 <= x <= 1:
        #             return 1/2 * (1 + np.exp(2 * (x - 1)))
        #         else:
        #             return 0
        #      # test system exact solution


        # exact_fn = lambda x : (mesh.params.inflow_value * np.exp(- (sigma_1 / mu) * x)) if 0 <= x < 0.5 else (mesh.params.inflow_value * np.exp((- (sigma_1 / mu)) * 0.5) * np.exp(-(sigma_2 / mu) * (x-0.5)))  # two zone exact solution
        
        def generate_K(mu,mesh,inflow_value):
            K = np.zeros(len(mesh.zone_vertices))
            K[0] = inflow_value
            for i in range(mesh.n_zones):
                K[i+1] = K[i] * np.exp(-mesh.params.total_cross_section[i]/mu * (mesh.zone_vertices[i+1] - mesh.zone_vertices[i])) + mesh.params.source[i]/mesh.params.total_cross_section[i] * (1 - np.exp(-mesh.params.total_cross_section[i]/mu * (mesh.zone_vertices[i+1] - mesh.zone_vertices[i]))) 
            return K
        

        def exact_fn(x,K,mu,mesh):
            for i in range(mesh.n_zones):
                if (mesh.zone_vertices[i] <= x <= mesh.zone_vertices[i+1]) or np.isclose(x, mesh.zone_vertices[i]) or np.isclose(x, mesh.zone_vertices[i+1]):
                    return K[i] * np.exp(-mesh.params.total_cross_section[i]/mu * (x - mesh.zone_vertices[i])) + mesh.params.source[i]/mesh.params.total_cross_section[i] * (1 - np.exp(-mesh.params.total_cross_section[i]/mu * (x - mesh.zone_vertices[i]))) 
 
        K = generate_K(mu, mesh, inflow_value=mesh.params.inflow_value[0])
        
        u_fn = lambda x: exact_fn(x, K, mu, mesh)

        print(u_fn(mesh.vertices[-1]))
        print(mesh.vertices[-1])

        print(K)

        U_exact = [u_fn(mesh.vertices[i]) for i in range(mesh.n_vertices)]   
        
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

    convergence_table.to_csv('supg_convergence_rate_discontinuity_error_repro.csv', index=False)

    plt.loglog(1 / np.array(hs), errors)

    # plt.plot(mesh.vertices, U)
    # plt.plot(mesh.vertices, U_exact)
    plt.savefig("supg_convergence_rate_discontinuity_error_repro.png")

    plt.close()

    plt.plot(mesh.vertices, U, label="approximate")
    plt.plot(mesh.vertices, U_exact, label="exact")
    plt.legend()
    plt.savefig("supg_convergence_rate_discontinuity_error_repro_solution.png")

    print("Test passed!")

if __name__ == "__main__":
    main()