from scipy.sparse import coo_matrix, csr_matrix
from scipy.integrate import quad
from .mesh import Mesh
import numpy as np


# def generate_sparsity(matrix: csr_matrix):
#     matrix = matrix.tocoo()
#     row = matrix.row
#     col = matrix.col
#     data = matrix.data
#     return row, col, data


def phi1hat(x):
    return x


def phi0hat(x):
    return 1 - x


def dphi1hat(x):
    return 1


def dphi0hat(x):
    return -1


gauss_points = [0.5 * (1 - 1 / np.sqrt(3)), 0.5 * (1 + 1 / np.sqrt(3))]
gauss_weights = [1 / 2, 1 / 2]


def assemble_matrix(mu: float, mesh: Mesh):
    row = []
    col = []
    data = []

    # Mass matrix
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

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    print(row)
    print(col)
    print(data)

    coo = coo_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))

    return csr_matrix(coo)


def compute_source_term(phi_index: int, cell_index: int, mesh: Mesh):
    if cell_index == phi_index:

        def integrand(x):
            return mesh.source[cell_index] * phi1hat(x) * mesh.h[cell_index]

    if cell_index == phi_index - 1:

        def integrand(x):
            return mesh.source[cell_index] * phi0hat(x) * mesh.h[cell_index]

    return quad(integrand, 0, 1)[0]


hat_fns = [phi0hat, phi1hat]


def assemble_rhs(mu, mesh: Mesh):
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

    return data


def mass_entry(i, j, gauss_matrix, mesh):
    if i > j:
        return (
            mesh.sigma_t[j] * np.dot(gauss_matrix[:][0], gauss_matrix[:][1]) * mesh.h[j]
        )
    elif i < j:
        return (
            mesh.sigma_t[i] * np.dot(gauss_matrix[:][0], gauss_matrix[:][1]) * mesh.h[i]
        )
    else:
        return (
            mesh.sigma_t[i]
            * (gauss_matrix[0][0] ** 2 + gauss_matrix[1][0] ** 2)
            * mesh.h[i]
            + mesh.sigma_t[i - 1]
            * (gauss_matrix[0][1] ** 2 + gauss_matrix[1][1] ** 2)
            * mesh.h[i - 1]
        )


def stiffness_entry(i, j, mu, mesh):
    if i > j:

        def integrand(x):
            return mu * phi0hat(x) * dphi1hat(x)
    elif i < j:

        def integrand(x):
            return mu * phi0hat(x) * dphi1hat(x)
    else:

        def integrand(x):
            return mu * phi1hat(x) * dphi1hat(x) + mu * phi0hat(x) * dphi0hat(x)

    return quad(integrand, 0, 1)[0]
