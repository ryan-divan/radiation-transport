import numpy as np
from scipy.sparse import csr_matrix
from .mesh import Mesh
from .utils import phi1hat, phi0hat, dphi1hat, dphi0hat, gauss_points, gauss_weights, hat_fns, dhat_fns

# Assemble the global matrix for the advection-diffusion problem 
def assemble_poisson_matrix(mesh: Mesh, robust: bool = True):
    
    mass_matrix = assemble_mass_matrix(mesh)

    s3_matrix = assemble_s3_matrix(mesh, robust)

    matrix = mass_matrix + s3_matrix

    omega = max(1, np.max(1 / (np.max(mesh.sigma_t) * mesh.h)))

    C = mesh.params.boundary_penalty

    matrix[0, 0] += C * omega 
    matrix[-1, -1] += C * omega 

    return matrix


def assemble_mass_matrix(mesh: Mesh):
    row = []
    col = []
    data = []

    for k in range(mesh.n_cells):
        h = mesh.h[k]
        per_cell_sigma_a = mesh.sigma_a[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (
                        h
                        * per_cell_sigma_a
                        * hat_fns[i_hat](gauss_points[x])
                        * hat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    mass_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return mass_matrix

def assemble_s3_matrix(mesh: Mesh, robust: bool = True):
    row = []
    col = []
    data = []

    max_sigma_t = np.max(mesh.sigma_t)
    epsilon = 1e-12

    for k in range(mesh.n_cells):
        h = mesh.h[k]
        per_cell_sigma_t = mesh.sigma_t[k] + epsilon * max_sigma_t if robust else mesh.sigma_t[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (
                        (1 / h)
                        * (1 / (per_cell_sigma_t * 3))
                        * dhat_fns[i_hat](gauss_points[x])
                        * dhat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    s3_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return s3_matrix

def assemble_poisson_rhs(mesh: Mesh):
    
    data = np.zeros((mesh.n_vertices))
    
    # Compute the q * v_h term
    for k in range(mesh.n_cells):
        h = mesh.h[k]
        q = mesh.source_per_cell[k]
        sigma_s_per_cell = mesh.sigma_s[k]
        r = 4 * np.pi * q

        for i_hat in range(0, 2):
            i_global = i_hat + k
            acc = 0
            for x in range(len(gauss_points)):
                acc += h * r * hat_fns[i_hat](gauss_points[x]) * gauss_weights[x]

            data[i_global] += acc
    
    return data