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
def assemble_matrix(mu: float, mesh: Mesh, boundary_condition: bool = True, use_supg: bool = True):
    
    C = 1 / (2 * np.abs(mu)) * 1 / (np.maximum(1, mesh.sigma_t * mesh.h))
    
    tau =  C * mesh.h if use_supg else np.zeros(mesh.n_cells)
    
    mass_matrix = assemble_mass_matrix(mu, mesh)

    stiffness_matrix = assemble_stiffness_matrix(mu, mesh)
    
    # matrix (3) is given by sigma_t * mu * tau * u_h * v_h'
    s3_matrix = assemble_s3_matrix(mu, mesh, tau)

    # matrix (4) is given by mu * mu * tau * u_h' * v_h'
    s4_matrix = assemble_s4_matrix(mu, mesh, tau)

    matrix = mass_matrix + stiffness_matrix + s3_matrix + s4_matrix
    # matrix = mass_matrix + stiffness_matrix #+ s3_matrix + s4_matrix

    if boundary_condition:
        if mu > 0:
            matrix[0, 0] += mu * hat_fns[0](0) + mu**2 * tau[0] * dhat_fns[0](0) * (1 / mesh.h[0])
            matrix[1, 0] += mu**2 * tau[0] * dhat_fns[1](0) * (1 / mesh.h[0])
        elif mu < 0:
            matrix[-1, -1] += mu * hat_fns[1](1) + mu**2 * tau[-1] * dhat_fns[1](1) * (1 / mesh.h[-1])
            matrix[-2, -1] += mu**2 * tau[-1] * dhat_fns[0](1) * (1 / mesh.h[-1])


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

def assemble_rhs(mu: float, mesh: Mesh, boundary_condition: bool = True, use_supg: bool = True):
    data = np.zeros((mesh.n_vertices))

    C = 1 / (2 * np.abs(mu)) * 1 / (np.maximum(1, mesh.sigma_t * mesh.h))
    
    tau =  C * mesh.h if use_supg else np.zeros(mesh.n_cells)
    
    # Compute the q * v_h term
    for k in range(mesh.n_cells):
        h = mesh.h[k]
        q = mesh.source_per_cell[k]

        for i_hat in range(0, 2):
            i_global = i_hat + k
            acc = 0
            for x in range(len(gauss_points)):
                acc += h * q * hat_fns[i_hat](gauss_points[x]) * gauss_weights[x]

            data[i_global] += acc

    if use_supg:
        # Compute the q * tau_k * mu * v_h'  term
        for k in range(mesh.n_cells):
            h = mesh.h[k]
            q = mesh.source_per_cell[k]
            per_cell_tau = tau[k]

            for i_hat in range(0, 2):
                i_global = i_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += q * per_cell_tau * mu * dhat_fns[i_hat](gauss_points[x]) * gauss_weights[x]

                data[i_global] += acc

    if boundary_condition:
        if mu > 0:
            # data[0] += mu * mesh.inflow_value
            data[0] += mu * mesh.inflow_value * hat_fns[0](0) + mu**2 * tau[0] * dhat_fns[0](0) * mesh.inflow_value * (1 / mesh.h[0])
            data[1] += mu**2 * tau[0] * dhat_fns[1](0) * mesh.inflow_value * (1 / mesh.h[0])
        elif mu < 0:
            data[-1] += mu * mesh.inflow_value * hat_fns[1](1) + mu**2 * tau[-1] * dhat_fns[1](1) * mesh.inflow_value * (1 / mesh.h[-1])
            data[-2] += mu**2 * tau[-1] * dhat_fns[0](1) * mesh.inflow_value * (1 / mesh.h[-1])

    return data

def assemble_s3_matrix(mu: float, mesh: Mesh, tau: np.array):
    row = []
    col = []
    data = []

    # mu^2 * tau * u_h' * v_h'
    for k in range(mesh.n_cells):
        per_cell_tau = tau[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (mu**2 * per_cell_tau
                        * dhat_fns[i_hat](gauss_points[x])
                        * dhat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x] / mesh.h[k]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    s3_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return s3_matrix

def assemble_s4_matrix(mu: float, mesh: Mesh, tau: np.array):
    row = []
    col = []
    data = []

    # sigma_t * mu * tau * u_h * v_h'
    for k in range(mesh.n_cells):
        per_cell_tau = tau[k]
        per_cell_sigma_t = mesh.sigma_t[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (mu * per_cell_tau * per_cell_sigma_t
                        * dhat_fns[i_hat](gauss_points[x])
                        * hat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    s4_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return s4_matrix
