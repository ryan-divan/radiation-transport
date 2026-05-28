import numpy as np
from scipy.special import roots_legendre

# Basis functions and derivatives on unit interval
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
    
hat_fns = [phi0hat, phi1hat]
dhat_fns = [dphi0hat, dphi1hat]


class AngularQuadrature:
    def __init__(self, order: int):
        self.order = order 
        self.angles, unnormalized_weights, sum = roots_legendre(order, mu=True)
        self.weights = (unnormalized_weights / sum) * 4 * np.pi
        self.verify()
    
    # input: array[num_ls, dimension]
    # output: array[dimension]
    def integrate(self, f):
        integral = f.T @ self.weights
        return integral
    
    def verify(self):
        assert self.order % 2 == 0, "Quadrature order must be even."