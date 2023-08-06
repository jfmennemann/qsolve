from qsolve.potentials.components_3d.lesanovsky_3d import eval_potential_lesanovsky_3d

import math

import torch

from scipy import constants


hbar = constants.hbar

mu_B = constants.physical_constants["Bohr magneton"][0]


class PotentialLesanovskyTiltX(object):

    def __init__(self, *, grid, units, device, parameters):

        self._units = units

        self._x_3d = torch.tensor(grid.x_3d / self._units.unit_length, device=device)
        self._y_3d = torch.tensor(grid.y_3d / self._units.unit_length, device=device)
        self._z_3d = torch.tensor(grid.z_3d / self._units.unit_length, device=device)

        self._hbar = hbar / self._units.unit_hbar
        self._mu_B = mu_B / self._units.unit_bohr_magneton

        self._g_F = -1/2
        self._m_F = -1
        self._m_F_prime = -1

        self._m_atom = parameters["m_atom"] / self._units.unit_mass

        self._omega_perp = 2.0 * math.pi * parameters["nu_perp"] / self._units.unit_frequency
        self._omega_para = 2.0 * math.pi * parameters["nu_para"] / self._units.unit_frequency
        self._omega_delta_detuning = 2.0 * math.pi * parameters["nu_delta_detuning"] / self._units.unit_frequency
        self._omega_trap_bottom = 2.0 * math.pi * parameters["nu_trap_bottom"] / self._units.unit_frequency
        self._omega_rabi_ref = 2.0 * math.pi * parameters["nu_rabi_ref"] / self._units.unit_frequency

        self._gamma_tilt_ref = parameters["gamma_tilt_ref"] * self._units.unit_length / self._units.unit_energy

    def compute_external_potential(self, t, u):

        _omega_rabi = u[0] * self._omega_rabi_ref

        V_lesanovsky = eval_potential_lesanovsky_3d(
            self._x_3d,
            self._y_3d,
            self._z_3d,
            self._g_F,
            self._m_F,
            self._m_F_prime,
            self._omega_perp,
            self._omega_para,
            self._omega_delta_detuning,
            self._omega_trap_bottom,
            _omega_rabi,
            self._hbar,
            self._mu_B,
            self._m_atom)

        V_tilt_x = -1.0 * u[1] * self._gamma_tilt_ref * self._x_3d

        return V_lesanovsky + V_tilt_x
