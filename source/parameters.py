import tomllib
import numpy as np


class ParameterHandler:
    def __init__(self, path: str):
        file = open(path, "rb")
        data = tomllib.load(file)
        self.length_per_zone = np.array(data["length_per_zone"])
        self.cells_per_zone = np.array(data["cells_per_zone"])
        self.total_cross_section = np.array(data["total_cross_section"])
        self.scattering_cross_section = np.array(data["scattering_cross_section"])
        self.source = np.array(data["source"])
        self.boundary_conditions = np.array(data["boundary_conditions"])
        self.verify()

    def verify(self):
        assert (
            len(self.length_per_zone)
            == len(self.cells_per_zone)
            == len(self.total_cross_section)
            == len(self.scattering_cross_section)
            == len(self.source)
        ), "All parameter arrays must have the same length."
        assert len(self.boundary_conditions) == 2, (
            "Boundary conditions must have exactly two values."
        )

    def __str__(self):
        return (
            f"\n Parameters:"
            f"\n length_per_zone={self.length_per_zone}, \n cells_per_zone={self.cells_per_zone},"
            f"\n total_cross_section={self.total_cross_section}, \n scattering_cross_section={self.scattering_cross_section},"
            f"\n \source={self.source}, \n boundary_conditions={self.boundary_conditions}"
        )
