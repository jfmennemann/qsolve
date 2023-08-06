import math

import torch

from scipy import constants


hbar = constants.hbar


class PotentialHarmonicXGaussianY(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x_2d = torch.tensor(grid.x_2d / self._units.unit_length, device=device)
        self._y_2d = torch.tensor(grid.y_2d / self._units.unit_length, device=device)

        self._hbar = hbar / self._units.unit_hbar

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_x = 2.0 * math.pi * parameters["nu_x"] / self._units.unit_frequency

        self._sigma_gaussian_y = parameters["sigma_gaussian_y"] / self._units.unit_length

        self._V_ref_gaussian_y = parameters["V_ref_gaussian_y"] / self._units.unit_energy

    def compute_external_potential(self, t, u):

        V_harmonic_x = 0.5 * self._m_atom * self._omega_x ** 2 * self._x_2d ** 2

        V_gaussian_y = torch.exp(-self._y_2d ** 2 / (2 * self._sigma_gaussian_y ** 2))

        amplitude_gaussian_y = u * self._V_ref_gaussian_y

        return V_harmonic_x + amplitude_gaussian_y * V_gaussian_y

