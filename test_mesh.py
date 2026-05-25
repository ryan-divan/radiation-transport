import sys
from source.parameters import ParameterHandler
from source.mesh import Mesh


def main():
    # check that the parameters file is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python3 test_mesh.py <parameter_file>")
        sys.exit(1)

    # load parameters
    params = ParameterHandler(sys.argv[1])

    # generate mesh
    mesh = Mesh(params)

    # print parameters and mesh information
    print(params)
    print(mesh)


if __name__ == "__main__":
    main()
