import numpy as np
from source.parameters import ParameterHandler
from source.mesh import Mesh
from scipy.sparse import csr_matrix
from source.poisson import assemble_poisson_matrix, assemble_poisson_rhs
from scipy.sparse.linalg import spsolve
import pandas as pd
import matplotlib.pyplot as plt

def main():


    params = ParameterHandler("test_params/poisson_benchmark.toml")

    # generate mesh
    mesh = Mesh(params)

    c_0 = 3 * mesh.sigma_t[0].item()
    c_1 = 3 * mesh.sigma_t[-1].item()
    q_0 = 4 * np.pi * mesh.params.source[0].item()
    q_1 = 4 * np.pi * mesh.params.source[-1].item()
    # q_0 = mesh.params.source[0].item()
    # q_1 = mesh.params.source[-1].item()
    print(c_0, c_1, q_0, q_1)


    lhs = np.array([[1, -1], [1, 1]])
    rhs = np.array([
        (c_0 * q_0) / 2 - (c_1 * q_1) / 2,   
        c_0 * q_0 + c_1 * q_1                
    ])

    ys = np.linalg.solve(lhs, rhs)
    print(ys)

    # def exact_fn(x):
    #     if 0 <= x <= 1:
    #         return ys[0] * x - ((c_0 * q_0) / 2) * x * x
    #     else:
    #         return ((c_1 * q_1) / 2) * ((2 - x) ** 2) - ys[1] * (2 - x)
    def exact_fn(x):
        if 0 <= x <= 1:
            return ys[0] * x - (c_0 * q_0 / 2) * x * x 
        else:
            return ys[1] * (2 - x) - (c_1 * q_1 / 2) * (2 - x) ** 2

    errors = []
    ps = []
    hs = []

    for _ in range(10):
        mesh = Mesh(params)
        D = assemble_poisson_matrix(mesh)
        Q = assemble_poisson_rhs(mesh)
        u = spsolve(D, Q)
        U_exact = [exact_fn(xs) for xs in mesh.vertices]
        hs.append(mesh.h[0])
        errors.append(np.sum(np.abs(U_exact - u) * mesh.lumped_mass))
        if len(errors) > 1:
            ps.append(np.log(errors[-2] / errors[-1]) / np.log(hs[-2] / hs[-1]))
        else:
            ps.append(None)
        params.cells_per_zone *= 2

    convergence_table = pd.DataFrame({
        'h': hs,
        'L1 Error Estimate': errors,
        'Empirical Convergence Rate p': ps
    })

    convergence_table.to_csv('convergence_rate_poisson_benchmark.csv', index=False)

    plt.loglog(1 / np.array(hs), errors)

    # plt.plot(mesh.vertices, U)
    # plt.plot(mesh.vertices, U_exact)
    plt.savefig("convergence_rate_poisson_benchmark.png")

    plt.close()

    plt.plot(mesh.vertices, u, label="approximate")
    plt.plot(mesh.vertices, U_exact, label="exact")
    plt.legend()
    plt.savefig("convergence_rate_poisson_benchmark_solution.png")

if __name__ == "__main__":
    main()