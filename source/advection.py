from scipy.sparse import coo_matrix, csr_matrix
from scipy.integrate import quad
from .mesh import Mesh
import numpy as np

# TODO: add connectivity array for higher-dimensional case
# def generate_connectivity_array(mesh: Mesh):
#     pass

# Basis functions and derivatives on unit interval
def phi1hat(x):
    return x


def phi0hat(x):
    return 1 - x


def dphi1hat(x):
    return 1


def dphi0hat(x):
    return -1

# Quadrature points and weights for 2-point Gauss quadrature on [0, 1]
gauss_points = [0.5 * (1 - 1 / np.sqrt(3)), 0.5 * (1 + 1 / np.sqrt(3))]
gauss_weights = [1 / 2, 1 / 2]

hat_fns = [phi0hat, phi1hat]
dhat_fns = [dphi0hat, dphi1hat]

# Assemble the global matrix for the advection-diffusion problem 
def assemble_matrix(mu: float, mesh: Mesh, boundary_condition: bool = True):
    
    mass_matrix = assemble_mass_matrix(mu, mesh)

    stiffness_matrix = assemble_stiffness_matrix(mu, mesh)

    matrix = mass_matrix + stiffness_matrix

    if boundary_condition:
        if mu > 0:
            matrix[0, 0] += mu
        elif mu < 0:
            matrix[-1, -1] += mu

    return matrix

# Assemble the mass matrix for the advection-diffusion problem.
# M_ij = integral of sigma_t * phi_i * phi_j dx, sigma_t is the total cross section.
# M is sparse, using CSR for efficiency.
# Loops over each cell, compute the local mass matrix for that cell, then add to M.
# Using 2-point Gauss quadrature for integrals.
# basis functions are the shifted to reference cell, evaluated at the quadrature points relative to reference cell. Factor of h because of this as integration factor from change of vars. 
def assemble_mass_matrix(mu: float, mesh: Mesh):
    row = []
    col = []
    data = []

    for k in range(mesh.n_cells):
        h = mesh.h[k]
        sigma_t = mesh.sigma_t[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (
                        h
                        * sigma_t
                        * hat_fns[i_hat](gauss_points[x])
                        * hat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    mass_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return mass_matrix

# Assemble the stiffness matrix for the advection-diffusion problem.
#   K_ij = integral mu * dphi_i/dx * dphi_j/dx dx, mu is the angle. Note again that basis are w.r.t reference cell.
# Loops over each cell, compute the local mass matrix for that cell, then add to S.
# Using 2-point Gauss quadrature for integrals.
# S is sparse matrix, and we will use the COO format to assemble it efficiently. 
# We will loop over each cell in the mesh, compute the local stiffness matrix for that cell, and then add it to the global stiffness matrix.
# The stiffness matrix will be added to the mass matrix to form the global matrix for the advection-diffusion problem.
def assemble_stiffness_matrix(mu: float, mesh: Mesh):

    row = []
    col = []
    data = []

    for k in range(mesh.n_cells):
        h = mesh.h[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (mu
                        * hat_fns[i_hat](gauss_points[x])
                        * dhat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    stiffness_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return stiffness_matrix

def assemble_rhs(mu: float, mesh: Mesh, boundary_condition: bool = True):
    data = np.zeros((mesh.n_vertices))

    # Mass matrix
    for k in range(mesh.n_cells):
        h = mesh.h[k]
        q = mesh.source[k]

        for i_hat in range(0, 2):
            i_global = i_hat + k
            acc = 0
            for x in range(len(gauss_points)):
                acc += h * q * hat_fns[i_hat](gauss_points[x]) * gauss_weights[x]

            data[i_global] += acc

    if boundary_condition:
        if mu > 0:
            data[0] += mu * mesh.inflow_value
        elif mu < 0:
            data[-1] += mu * mesh.inflow_value

    return data

