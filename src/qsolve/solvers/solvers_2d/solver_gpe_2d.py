import torch

import scipy

import sys

import math

from qsolve.core import qsolve_core

# import qsolve_core

from qsolve.units import Units


class SolverGPE2D(object):

    def __init__(self, *, grid, m_atom, a_s, omega_z, seed=0, device='cpu', num_threads_cpu=1):

        # -----------------------------------------------------------------------------------------
        print("Python version:")
        print(sys.version)
        print()
        print("PyTorch version:")
        print(torch.__version__)
        print()
        # -----------------------------------------------------------------------------------------

        torch.manual_seed(seed)

        torch.set_num_threads(num_threads_cpu)

        self._device = torch.device(device)

        self._units = Units.solver_units(m_atom, dim=2)

        # -----------------------------------------------------------------------------------------
        self._hbar = scipy.constants.hbar / self._units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self._m_atom = m_atom / self._units.unit_mass
        self._a_s = a_s / self._units.unit_length

        _omega_z = omega_z / self._units.unit_frequency

        _g_3d = 4.0 * math.pi * self._hbar ** 2 * self._a_s / self._m_atom

        _a_z = math.sqrt(self._hbar / (self._m_atom * _omega_z))

        self._g = _g_3d / (math.sqrt(2 * math.pi) * _a_z)

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        self._x = torch.tensor(grid.x / self._units.unit_length, device=self._device)
        self._y = torch.tensor(grid.y / self._units.unit_length, device=self._device)

        self._x_min = grid.x_min / self._units.unit_length
        self._x_max = grid.x_max / self._units.unit_length

        self._y_min = grid.y_min / self._units.unit_length
        self._y_max = grid.y_max / self._units.unit_length

        self._Lx = grid.Lx / self._units.unit_length
        self._Ly = grid.Ly / self._units.unit_length

        self._Jx = grid.Jx
        self._Jy = grid.Jy

        self._dx = grid.dx / self._units.unit_length
        self._dy = grid.dy / self._units.unit_length

        self._index_center_x = grid.index_center_x
        self._index_center_y = grid.index_center_y

        self._x_2d = torch.tensor(grid.x_2d / self._units.unit_length, device=self._device)
        self._y_2d = torch.tensor(grid.y_2d / self._units.unit_length, device=self._device)

        self._compute_external_potential = None
        self._V = None

        self._psi = None

        self._p = {
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "k_B": self._k_B,
            "m_atom": self._m_atom
        }

    def init_external_potential(self, compute_external_potential, parameters_potential):

        self._compute_external_potential = compute_external_potential

        for key, p in parameters_potential.items():

            if type(p) is not tuple:

                _value = p

            else:

                value = p[0]
                unit = p[1]

                if unit == 'm':
                    _value = value / self._units.unit_length
                elif unit == 's':
                    _value = value / self._units.unit_time
                elif unit == 'Hz':
                    _value = value / self._units.unit_frequency
                elif unit == 'J':
                    _value = value / self._units.unit_energy
                elif unit == 'J/m':
                    _value = value * self._units.unit_length / self._units.unit_energy
                else:
                    raise Exception('unknown unit')

            self._p[key] = _value

    def set_external_potential(self, *, t, u):

        _t = t / self._units.unit_time

        self._V = self._compute_external_potential(self._x_2d, self._y_2d, t, u, self._p)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau=True, return_residuals=False):

        _tau = tau / self._units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core.ground_state_gpe_2d(
            self._V,
            self._dx,
            self._dy,
            _tau,
            adaptive_tau,
            n_iter,
            n_atoms,
            self._hbar,
            self._m_atom,
            self._g)

        if return_residuals:

            return self._units.unit_wave_function * _psi_0.cpu().numpy(), vec_res, vec_iter

        else:

            return self._units.unit_wave_function * _psi_0.cpu().numpy()

    def propagate_gpe(self, *, times, u_of_times, n_start, n_inc, mue_shift=0.0):

        _times = times / self._units.unit_time
        _dt = _times[1] - _times[0]

        _mue_shift = mue_shift / self._units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            _t = _times[n]

            if u_of_times.ndim > 1:

                u = 0.5 * (u_of_times[:, n] + u_of_times[:, n + 1])

            else:

                u = 0.5 * (u_of_times[n] + u_of_times[n + 1])

            self._V = self._compute_external_potential(self._x_2d, self._y_2d, _t, u, self._p)

            self._psi = qsolve_core.propagate_gpe_2d(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                _dt,
                _mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def compute_n_atoms(self):
        return qsolve_core.n_atoms_2d(self._psi, self._dx, self._dy)

    def compute_chemical_potential(self):

        _mue = qsolve_core.chemical_potential_gpe_2d(
            self._psi, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _mue

    def compute_total_energy(self):

        _E = qsolve_core.total_energy_gpe_2d(
            self._psi, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _E

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core.kinetic_energy_lse_2d(self._psi, self._dx, self._dy, self._hbar, self._m_atom)

        return self._units.unit_energy * _E_kinetic

    def compute_potential_energy(self):

        _E_potential = qsolve_core.potential_energy_lse_2d(self._psi, self._V, self._dx, self._dy)

        return self._units.unit_energy * _E_potential

    def compute_interaction_energy(self):

        _E_interaction = qsolve_core.interaction_energy_gpe_2d(self._psi, self._dx, self._dy, self._g)

        return self._units.unit_energy * _E_interaction

    @property
    def x(self):
        return self._units.unit_length * self._x.cpu().numpy()

    @property
    def y(self):
        return self._units.unit_length * self._y.cpu().numpy()

    @property
    def index_center_x(self):
        return self._index_center_x

    @property
    def index_center_y(self):
        return self._index_center_y

    @property
    def V(self):
        return self._units.unit_energy * self._V.cpu().numpy()

    @property
    def psi(self):
        return self._units.unit_wave_function * self._psi.cpu().numpy()

    @psi.setter
    def psi(self, value):
        self._psi = torch.tensor(value / self._units.unit_wave_function, device=self._device)
