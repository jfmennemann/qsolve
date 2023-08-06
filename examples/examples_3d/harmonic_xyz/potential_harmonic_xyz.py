import math

import torch

from scipy import constants


hbar = constants.hbar


class PotentialHarmonicXYZ(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x_3d = torch.tensor(grid.x_3d / self._units.unit_length, device=device)
        self._y_3d = torch.tensor(grid.y_3d / self._units.unit_length, device=device)
        self._z_3d = torch.tensor(grid.z_3d / self._units.unit_length, device=device)

        self._hbar = hbar / self._units.unit_hbar

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_x = 2.0 * math.pi * parameters["nu_x"] / self._units.unit_frequency
        self._omega_y = 2.0 * math.pi * parameters["nu_y"] / self._units.unit_frequency
        self._omega_z = 2.0 * math.pi * parameters["nu_z"] / self._units.unit_frequency

    def compute_external_potential(self, t, u):

        return 0.5 * self._m_atom * (self._omega_x**2 * self._x_3d**2
                                     + self._omega_y**2 * self._y_3d**2 + self._omega_z**2 * self._z_3d**2)
