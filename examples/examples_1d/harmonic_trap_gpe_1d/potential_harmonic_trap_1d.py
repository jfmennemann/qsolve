# def compute_external_potential(x, t, u, p):
#
#     nu_start = p["nu_start"]
#     nu_final = p["nu_final"]
#
#     m_atom = p["m_atom"]
#
#     omega_start = 2 * np.pi * nu_start
#     omega_final = 2 * np.pi * nu_final
#
#     u = u[0]
#
#     omega = omega_start + u * (omega_final - omega_start)
#
#     V = 0.5 * m_atom * omega ** 2 * x ** 2
#
#     return V


import numpy as np

import math

import torch

from scipy import constants


hbar = constants.hbar


class PotentialHarmonicTrap1D(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x = torch.tensor(grid.x / self._units.unit_length, device=device)

        self._hbar = hbar / self._units.unit_hbar

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_start = 2 * np.pi * parameters["nu_start"] / self._units.unit_frequency
        self._omega_final = 2 * np.pi * parameters["nu_final"] / self._units.unit_frequency

        # self._omega_x = 2.0 * math.pi * parameters["nu_x"] / self._units.unit_frequency

        # self._sigma_gaussian_y = parameters["sigma_gaussian_y"] / self._units.unit_length

        # self._V_ref_gaussian_y = parameters["V_ref_gaussian_y"] / self._units.unit_energy

    def compute_external_potential(self, t, u):

        u = u[0]

        omega = self._omega_start + u * (self._omega_final - self._omega_start)

        V = 0.5 * self._m_atom * omega ** 2 * self._x ** 2

        return V
