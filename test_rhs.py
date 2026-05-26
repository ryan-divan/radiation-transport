import sys
import numpy as np
from source.parameters import ParameterHandler
from source.mesh import Mesh
from source.advection import assemble_rhs


def main():
    params = ParameterHandler("test_params/very_simple_params.toml")

    # generate mesh
    mesh = Mesh(params)

    rhs = assemble_rhs(1.0, mesh, boundary_condition=False, use_supg=False)
    assert np.allclose(rhs, [0.25, 0.5, 0.25, 0.0, 0.0])

    print(rhs)

    print("Test passed!")

if __name__ == "__main__":
    main()