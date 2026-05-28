import tomllib
import numpy as np


class ParameterHandler:
    def __init__(self, path: str):
        with open(path, "rb") as file:
            data = tomllib.load(file)
            self.length_per_zone = np.array(data["length_per_zone"])
            self.cells_per_zone = np.array(data["cells_per_zone"])
            self.total_cross_section = np.array(data["total_cross_section"])
            self.scattering_cross_section = np.array(data["scattering_cross_section"])
            self.source = np.array(data["source"])
            self.inflow_value = data["inflow_value"]
            self.number_of_directions = data["number_of_directions"]
            self.tol = data["tol"]
            self.iter_max = data["iter_max"]
            self.use_dsa = data.get("use_dsa", True)
            self.boundary_penalty = data.get("boundary_penalty", 1 / self.tol)
        self.verify()

    def verify(self):
        assert (
            len(self.length_per_zone)
            == len(self.cells_per_zone)
            == len(self.total_cross_section)
            == len(self.scattering_cross_section)
            == len(self.source)
        ), "All parameter arrays must have the same length."
        assert isinstance(self.inflow_value, (int, float)), (
            "Inflow value must be a scalar."
        )
        assert isinstance(self.number_of_directions, int), (
            "Number of directions must be an integer."
        )

    def __str__(self):
        return (
            f"\n Parameters:"
            f"\n length_per_zone={self.length_per_zone}, \n cells_per_zone={self.cells_per_zone},"
            f"\n total_cross_section={self.total_cross_section}, \n scattering_cross_section={self.scattering_cross_section},"
            f"\n source={self.source}, \n inflow_value={self.inflow_value}, \n number_of_directions={self.number_of_directions}"
        )
