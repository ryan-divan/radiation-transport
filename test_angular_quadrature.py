import numpy as np
from source.utils import AngularQuadrature

def main():
    for order in range(1, 10):
        quadrature = AngularQuadrature(order)
        n = 2 * order - 1
        p = np.arange(n )
        c =  np.random.random(size = n)
        angles = quadrature.angles
        f = np.array([np.sum(c * np.power(angles[i], p)) for i in range(len(angles))])
        exact_int = 2 * np.pi * sum(c * (1 - (-1)**(p + 1)) / (p + 1))
        integral = quadrature.integrate(f)
        assert np.isclose(integral, exact_int), f"Failed for order {order} and n {n}: got {integral}, expected {exact_int}"
    print("All tests passed!")

if __name__ == "__main__":
    main()