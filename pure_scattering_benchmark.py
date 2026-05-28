# %%
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
    params = ParameterHandler("test_params/scattering_benchmark.toml")
    params.total_cross_section[0] = 10
    for i in range(10):
        sigma_s = params.total_cross_section[0] / (2 ** i)

        params.scattering_cross_section[0] = sigma_s

        mesh = Mesh(params)
        
        phi_output, psi_outputs, num_iters = solve_source_iteration(mesh, return_iter=True)
        print(f"sigma_s={sigma_s}, sigma_t={params.total_cross_section[0]}, ratio={sigma_s / params.total_cross_section[0]}")
        print(f"num_iters={num_iters}")

        


    # plt.plot(mesh.vertices, phi_output)
    # plt.savefig("radiation_iteration_phi_scattering_benchmark.png")
    # plt.close()

    # print(psi_outputs.shape)
    # plt.plot(mesh.vertices, psi_outputs[0])
    # plt.savefig("radiation_iteration_psi_scattering_benchmark.png")
    # plt.close()


    # print(psi_outputs.shape)

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # quadrature = AngularQuadrature(mesh.params.number_of_directions)

    # num_angles, num_vertices = psi_outputs.shape
    # X, Y = np.meshgrid(mesh.vertices, quadrature.angles)

    # ax.plot_surface(X, Y, psi_outputs, cmap='viridis', edgecolor='none')

    # ax.set_xlabel('x (spatial)')
    # ax.set_ylabel('Angle index')
    # ax.set_zlabel('ψ')
    # ax.set_title('Angular Flux - Scattering Benchmark')

    # plt.savefig("radiation_iteration_psi_scattering_benchmark_3d.png", dpi=150, bbox_inches='tight')
    # plt.close()

    
    # print(num_iters)

if __name__ == "__main__":
    main()
# %%
