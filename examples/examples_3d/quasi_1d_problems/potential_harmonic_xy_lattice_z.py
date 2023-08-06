import math

import torch

from scipy import constants


hbar = constants.hbar


class PotentialHarmonicXYLatticeZ(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x_3d = torch.tensor(grid.x_3d / self._units.unit_length, device=device)
        self._y_3d = torch.tensor(grid.y_3d / self._units.unit_length, device=device)
        self._z_3d = torch.tensor(grid.z_3d / self._units.unit_length, device=device)

        self._Lz = grid.Lz / self._units.unit_length

        self._hbar = hbar / self._units.unit_hbar

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_x = 2.0 * math.pi * parameters["nu_x"] / self._units.unit_frequency
        self._omega_y = 2.0 * math.pi * parameters["nu_y"] / self._units.unit_frequency

        self._V_lattice_z_max = parameters["V_lattice_z_max"] / self._units.unit_energy

        self._m = parameters["V_lattice_z_m"]

    def compute_external_potential(self, t, u):

        _V_harmonic_xy = 0.5 * self._m_atom * (self._omega_x ** 2 * self._x_3d ** 2 + self._omega_y ** 2 * self._y_3d ** 2)

        _V_lattice_z = 0.5 * (torch.cos(2.0 * math.pi * self._m * self._z_3d / self._Lz) + 1.0)

        return _V_harmonic_xy + u * self._V_lattice_z_max * _V_lattice_z
