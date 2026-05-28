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

            self.number_of_directions = data.get("number_of_directions", 1)
            self.inflow_value = data["inflow_value"]
            if not isinstance(self.inflow_value, list):
                self.inflow_value = [self.inflow_value for _ in range(self.number_of_directions)]

            self.tol = data.get("tol", 1e-8)
            self.iter_max = data.get("iter_max", 1000)
            self.use_dsa = data.get("use_dsa", True)
            self.tau_ap = data.get("tau_ap", True)
            self.boundary_penalty = data.get("boundary_penalty", 1 / 4)

        self.verify()

    def verify(self):
        assert (
            len(self.length_per_zone)
            == len(self.cells_per_zone)
            == len(self.total_cross_section)
            == len(self.scattering_cross_section)
            == len(self.source)
        ), "All parameter arrays must have the same length."
        assert all([isinstance(elem, (int, float)) for elem in self.inflow_value]), (
            "Inflow value must be a scalar."
        )
        assert isinstance(self.number_of_directions, int), (
            "Number of directions must be an integer."
        )
        assert self.number_of_directions == len(self.inflow_value), (
            "Length of inflow_value array must match number_of_directions."
        )

    def __str__(self):
        return (
            f"\n Parameters:"
            f"\n length_per_zone={self.length_per_zone}, \n cells_per_zone={self.cells_per_zone},"
            f"\n total_cross_section={self.total_cross_section}, \n scattering_cross_section={self.scattering_cross_section},"
            f"\n source={self.source}, \n inflow_value={self.inflow_value}, \n number_of_directions={self.number_of_directions}"
        )
