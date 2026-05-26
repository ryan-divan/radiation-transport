import numpy as np
from .parameters import ParameterHandler
from scipy.integrate import quad


class Mesh:
    def __init__(self, params: ParameterHandler):
        self.length_per_zone = params.length_per_zone
        self.cells_per_zone = params.cells_per_zone
        self.total_cross_section = params.total_cross_section
        self.scattering_cross_section = params.scattering_cross_section
        self.source = params.source
        self.inflow_value = params.inflow_value

        self.total_length = np.sum(self.length_per_zone)
        self.n_cells = np.sum(self.cells_per_zone)
        self.n_vertices = np.sum(self.cells_per_zone) + 1
        self.n_zones = len(self.length_per_zone)
        self.h = np.repeat(self.length_per_zone / self.cells_per_zone, self.cells_per_zone)
        self.zone_vertices = np.cumsum(np.concatenate(([0], self.length_per_zone)))
        self.vertices = np.cumsum(np.concatenate(([0], self.h)))
        self.lumped_mass = self.compute_lumped_mass_array()
        self.material_id = np.repeat(np.arange(len(self.length_per_zone)), self.cells_per_zone)
        self.sigma_t = np.repeat(self.total_cross_section, self.cells_per_zone)
        self.sigma_s = np.repeat(self.scattering_cross_section, self.cells_per_zone)
        self.sigma_a = self.sigma_t - self.sigma_s

        self.source_per_cell = np.repeat(self.source, self.cells_per_zone)

    def compute_lumped_mass_array(self):
        return [self.compute_lumped_mass(i) for i in range(len(self.vertices))]

    def compute_lumped_mass(self, i):
        def phi(x):
            if i > 0 and self.vertices[i - 1] <= x <= self.vertices[i]:
                return (x - self.vertices[i - 1]) / (
                    self.vertices[i] - self.vertices[i - 1]
                )
            elif (
                i < len(self.vertices) and self.vertices[i] <= x <= self.vertices[i + 1]
            ):
                return (self.vertices[i + 1] - x) / (
                    self.vertices[i + 1] - self.vertices[i]
                )
            else:
                return 0.0

        if i == 0:
            return quad(phi, self.vertices[0], self.vertices[1])[0]
        elif i == len(self.vertices) - 1:
            return quad(phi, self.vertices[-2], self.vertices[-1])[0]
        else:
            return quad(phi, self.vertices[i - 1], self.vertices[i + 1])[0]

    def __str__(self):
        return (
            f"\n Mesh:"
            f"\n total_length={self.total_length}, \n n_cells={self.n_cells}, "
            f"\n n_vertices={self.n_vertices}, \n h={self.h}, \n vertices={self.vertices}, "
            f"\n lumped_mass={self.lumped_mass}, \n material_id={self.material_id}, "
            f"\n sigma_t={self.sigma_t}, \n sigma_s={self.sigma_s}, \n sigma_a={self.sigma_a},"
            f"\n source={self.source}, \n inflow_value={self.inflow_value}"
        )
