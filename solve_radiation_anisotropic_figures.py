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

    CELLS = [10000] # [100, 1000, 10000]
    
    params = ParameterHandler(f"test_params/grazing_sigma100_20K.toml")
    for cell in CELLS:
        params.cells_per_zone = cell
        mesh = Mesh(params)
        phi_output, psi_outputs, num_iters = solve_source_iteration(mesh, return_iter=True)
        phi_output = phi_output / (2 * np.pi)
        plt.plot(mesh.vertices, phi_output, label="Points per zone: " + str(cell))

    plt.xlabel("x")
    plt.ylabel("Scalar Flux")
    plt.title("Anisotropic Scattering with Grazing Angles")

    plt.show()

    plt.legend()

    # plt.savefig(f"radiation_iteration_grazing_sigma100_comparison.png", dpi=150)

    plt.close()

    # print(psi_outputs.shape)
    # plt.plot(mesh.vertices, psi_outputs[0])
    # # plt.savefig(f"radiation_iteration_psi_{filename}.png")
    # # plt.close()

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # quadrature = AngularQuadrature(mesh.params.number_of_directions)

    # num_angles, num_vertices = psi_outputs.shape
    # X, Y = np.meshgrid(mesh.vertices, quadrature.angles)

    # ax.plot_surface(X, Y, psi_outputs, cmap='viridis', edgecolor='none')

    # ax.set_xlabel('x (spatial)')
    # ax.set_ylabel('mu (angular)')
    # ax.set_zlabel('ψ (angular flux)')
    # ax.set_title('Angular Flux - Reed\'s Problem')

    # plt.savefig(f"radiation_iteration_{filename}_3d.png", dpi=150)
    # plt.close()

    # print("Number of iterations:", num_iters)

if __name__ == "__main__":
    main()
# %%
