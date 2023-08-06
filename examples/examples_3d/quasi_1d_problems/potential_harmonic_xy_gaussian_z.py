import math

import torch

from scipy import constants


hbar = constants.hbar


class PotentialHarmonicXYGaussianZ(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x_3d = torch.tensor(grid.x_3d / self._units.unit_length, device=device)
        self._y_3d = torch.tensor(grid.y_3d / self._units.unit_length, device=device)
        self._z_3d = torch.tensor(grid.z_3d / self._units.unit_length, device=device)

        self._hbar = hbar / self._units.unit_hbar

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_x = 2.0 * math.pi * parameters["nu_x"] / self._units.unit_frequency
        self._omega_y = 2.0 * math.pi * parameters["nu_y"] / self._units.unit_frequency

        self._V_ref_gaussian_z = parameters['V_ref_gaussian_z'] / self._units.unit_energy
        self._sigma_gaussian_z = parameters['sigma_gaussian_z'] / self._units.unit_length

    def compute_external_potential(self, t, u):

        _V_harmonic_xy = 0.5 * self._m_atom * (self._omega_x ** 2 * self._x_3d ** 2 + self._omega_y ** 2 * self._y_3d ** 2)

        _V_gaussian_z = u * self._V_ref_gaussian_z * torch.exp(-self._z_3d ** 2 / (2 * self._sigma_gaussian_z ** 2))

        return _V_harmonic_xy + _V_gaussian_z
