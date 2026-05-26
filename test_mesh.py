import sys
import numpy as np
from source.parameters import ParameterHandler
from source.mesh import Mesh


def main():
    params = ParameterHandler("test_params/very_simple_params.toml")

    # generate mesh
    mesh = Mesh(params)

    print(mesh)
    # print parameters and mesh information

    assert np.isclose(mesh.total_length, 2.0) and mesh.n_cells == 4 and mesh.n_vertices == 5
    assert np.allclose(mesh.h, [0.5, 0.5, 0.5, 0.5])
    assert np.allclose(mesh.vertices, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.isclose(mesh.lumped_mass[0], 0.25) and np.isclose(mesh.lumped_mass[-1], 0.25)
    assert np.isclose(mesh.lumped_mass[1], 0.5) and np.isclose(mesh.lumped_mass[2], 0.5) and np.isclose(mesh.lumped_mass[3], 0.5)
    assert mesh.material_id[0] == 0 and mesh.material_id[1] == 0
    assert mesh.material_id[2] == 1 and mesh.material_id[3] == 1

    print("Test passed!")

if __name__ == "__main__":
    main()