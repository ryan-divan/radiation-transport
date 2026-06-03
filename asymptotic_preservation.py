import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_matrix, assemble_rhs
from source.discrete_ordinates import solve_source_iteration
from source.utils import AngularQuadrature
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def main():
    if len(sys.argv) < 2:
        print("Usage: python asymptotic_preservation.py test_params/<filename>.toml")
        return
    
    filename = sys.argv[1]

    params = ParameterHandler(f"test_params/{filename}.toml")

    mesh = Mesh(params)

    epsilons = [0, 0.25, 0.5, 0.9, 1]

    iters = []

    for eps in epsilons:
        mesh.sigma_s = eps * mesh.sigma_t 
        mesh.sigma_a = (1 - eps) * mesh.sigma_t
        phi_output, _, num_iters = solve_source_iteration(mesh, return_iter=True)
        phi_output = phi_output / (2 * np.pi)
        iters.append(num_iters)
        plt.plot(mesh.vertices, phi_output, label=f"epsilon={eps}")
        plt.xlabel("x")
        plt.ylabel("Scalar Flux")
        plt.title(f"Scattering Benchmark - epsilon={eps}, iters={num_iters}")
        plt.legend()
        plt.savefig(f"asymptotic_preservation_{filename}_epsilon={eps}.png", dpi=150)
        plt.close()

    pandas_df = pd.DataFrame({"epsilon": epsilons, "iterations": iters})
    pandas_df.to_csv(f"asymptotic_preservation_{filename}_iterations.csv", index=False)
        
if __name__ == "__main__":
    main()