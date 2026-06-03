import numpy as np
from source.mesh import Mesh
from .utils import phi1hat, phi0hat, dphi1hat, dphi0hat, gauss_points, gauss_weights, hat_fns, dhat_fns, AngularQuadrature, tau_fn
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from .advection import assemble_matrix, assemble_rhs
from .poisson import assemble_poisson_matrix, assemble_poisson_rhs
from scipy.sparse.linalg import spsolve, gmres, LinearOperator

gauss_points = [0.5 * (1 - 1 / np.sqrt(3)), 0.5 * (1 + 1 / np.sqrt(3))]
gauss_weights = [1 / 2, 1 / 2]


def assemble_sigma_matrix(mu: float, mesh: Mesh):
    row = []
    col = []
    data = []

    tau = tau_fn(mesh, mu)

    # (sigma_s / 4pi) * u_h * v_h
    for k in range(mesh.n_cells):
        h = mesh.h[k]
        per_cell_sigma_s = mesh.sigma_s[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (
                        h
                        * (per_cell_sigma_s / (4 * np.pi))
                        * hat_fns[i_hat](gauss_points[x])
                        * hat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)


    # (sigma_s / 4pi) * mu * tau * u_h * v_h'
    for k in range(mesh.n_cells):
        per_cell_tau = tau[k]
        per_cell_sigma_s = mesh.sigma_s[k]
        for i_hat in range(0, 2):
            for j_hat in range(0, 2):
                i_global = i_hat + k
                j_global = j_hat + k
                acc = 0
                for x in range(len(gauss_points)):
                    acc += (
                        (per_cell_sigma_s / (4 * np.pi))
                        * per_cell_tau  
                        * mu 
                        * dhat_fns[i_hat](gauss_points[x])
                        * hat_fns[j_hat](gauss_points[x])
                        * gauss_weights[x]
                    )

                row.append(i_global)
                col.append(j_global)
                data.append(acc)

    sigma_matrix = csr_matrix((data, (row, col)), shape=(mesh.n_vertices, mesh.n_vertices))
    
    return sigma_matrix

def apply_operator_A(mesh: Mesh, phi: np.array, mus: np.array, sigma_matrices: np.array, lhs_matrices: np.array, quadrature: AngularQuadrature):
    l = len(mus)

    psis = []
    for i in range(l):
        mu = mus[i]
        # 1. compute rhs < sigma_s / 4pi * phi, v >
        rhs = sigma_matrices[i] @ phi
        # 2. solve eq. 3.5b using sparse solver to get A @ phi
        psi = sp.linalg.spsolve(lhs_matrices[i], rhs)
        psis.append(psi)

    # psis: list[np.array[dimension]]
    # psis_stacked: np.array[num_ls, dimension]
    psis_stacked = np.stack(psis)
    quad_result = quadrature.integrate(psis_stacked)
    
    return phi - quad_result, psis_stacked

def l1_norm(mesh: Mesh, coefficients: np.array):
    return np.sum(np.abs(coefficients) * mesh.lumped_mass)


def assemble_dsa_rhs(mesh: Mesh, r: np.array):
    data = np.zeros((mesh.n_vertices))
    
    # Compute the sigma_s * r * v_h term
    for k in range(mesh.n_cells):
        h = mesh.h[k]
        r_per_cell = r[k:k+2]
        r_func = lambda x: r_per_cell[0] * hat_fns[0](x) + r_per_cell[1] * hat_fns[1](x)
        per_cell_sigma_s = mesh.sigma_s[k]
        for i_hat in range(0, 2):
            i_global = i_hat + k
            acc = 0
            for x in range(len(gauss_points)):
                acc += (
                    h
                    * per_cell_sigma_s
                    * r_func(gauss_points[x])
                    * hat_fns[i_hat](gauss_points[x])
                    * gauss_weights[x]
                )

            data[i_global] += acc

    return data

def solve_source_iteration(mesh: Mesh, return_iter: bool = False, use_gmres: bool = False):
    quadrature = AngularQuadrature(mesh.params.number_of_directions)
    mus = quadrature.angles
    use_dsa = mesh.params.use_dsa

    psi_stars = []
    for i in range(len(mus)):
        mu = mus[i]
        M = assemble_matrix(mu, mesh)
        Q = assemble_rhs(mu, inflow_value=mesh.params.inflow_value[i], mesh=mesh)

        psi_star = sp.linalg.spsolve(M, Q)
        psi_stars.append(psi_star)
    
    psi_stars = np.stack(psi_stars)
    phi_star = quadrature.integrate(psi_stars)

    phi = np.zeros(mesh.n_vertices)

    sigma_matrices = [assemble_sigma_matrix(mu, mesh) for mu in mus]
    lhs_matrices = [assemble_matrix(mu, mesh) for mu in mus]

    A_phi, _ = apply_operator_A(mesh, phi, mus, sigma_matrices, lhs_matrices, quadrature)
    r = phi_star - A_phi

    phi_star_norm = l1_norm(mesh, phi_star)

    M = assemble_poisson_matrix(mesh) if use_dsa else None # poisson matrix for DSA preconditioning

    if use_gmres:
        A_op = LinearOperator((mesh.n_vertices, mesh.n_vertices), matvec=lambda x: apply_operator_A(mesh, x, mus, sigma_matrices, lhs_matrices, quadrature)[0])

        reference_scale = np.linalg.norm(phi_star) + 1e-13

        M_op = LinearOperator((mesh.n_vertices, mesh.n_vertices), matvec=lambda x: x + sp.linalg.spsolve(M, assemble_dsa_rhs(mesh, x))) if use_dsa else None

        history = []
        def callback(residual):
            rel_err = residual / reference_scale
            history.append(rel_err)
        
        phi, info = gmres(A_op, phi_star, x0=phi, rtol=mesh.params.tol, maxiter=mesh.params.iter_max, M=M_op, callback=callback, callback_type='legacy')

        _, psis_stacked = apply_operator_A(mesh, phi, mus, sigma_matrices, lhs_matrices, quadrature)
        final_psis = psis_stacked + psi_stars

        k = len(history)

        if return_iter:
            return phi, final_psis, k 


    else:
        k = 0
        while l1_norm(mesh, r) / phi_star_norm > mesh.params.tol and k <= mesh.params.iter_max:
            phi += r + sp.linalg.spsolve(M, assemble_dsa_rhs(mesh, r)) if use_dsa else r
            A_phi, psis_stacked = apply_operator_A(mesh, phi, mus, sigma_matrices, lhs_matrices, quadrature)
            r = phi_star - A_phi
            k += 1

        final_psis = psis_stacked + psi_stars
        if return_iter:
            return phi, final_psis, k    

        return phi, final_psis