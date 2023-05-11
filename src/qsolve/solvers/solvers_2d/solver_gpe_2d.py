import torch

import scipy

import numpy as np

import sys

import math

from qsolve.core import qsolve_core_gpe_2d

from qsolve.primes import get_prime_factors

from qsolve.units import Units


class SolverGPE2D(object):

    def __init__(self, *, m_atom, a_s, omega_z, seed=0, device='cpu', num_threads_cpu=1):

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

        _g_3d = 4.0 * scipy.constants.pi * self._hbar ** 2 * self._a_s / self._m_atom

        _a_z = math.sqrt(self._hbar / (self._m_atom * _omega_z))

        self._g = _g_3d / (math.sqrt(2 * math.pi) * _a_z)

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        self._x = None
        self._y = None

        self._x_min = None
        self._x_max = None

        self._y_min = None
        self._y_max = None

        self._Lx = None
        self._Ly = None

        self._Jx = None
        self._Jy = None

        self._dx = None
        self._dy = None

        self._index_center_x = None
        self._index_center_y = None

        self._x_2d = None
        self._y_2d = None

        self._compute_external_potential = None
        self._V = None

        self._psi = None



        self.u_of_times = None

        self._p = {
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "k_B": self._k_B,
            "m_atom": self._m_atom
        }

    def init_grid(self, **kwargs):

        self._x_min = kwargs['x_min'] / self._units.unit_length
        self._x_max = kwargs['x_max'] / self._units.unit_length

        self._y_min = kwargs['y_min'] / self._units.unit_length
        self._y_max = kwargs['y_max'] / self._units.unit_length

        self._Jx = kwargs['Jx']
        self._Jy = kwargs['Jy']

        prime_factors_Jx = get_prime_factors(self._Jx)
        prime_factors_Jy = get_prime_factors(self._Jy)

        assert (np.max(prime_factors_Jx) < 11)
        assert (np.max(prime_factors_Jy) < 11)

        assert (self._Jx % 2 == 0)
        assert (self._Jy % 2 == 0)

        _x = np.linspace(self._x_min, self._x_max, self._Jx, endpoint=False)
        _y = np.linspace(self._y_min, self._y_max, self._Jy, endpoint=False)

        self._index_center_x = np.argmin(np.abs(_x))
        self._index_center_y = np.argmin(np.abs(_y))

        assert (np.abs(_x[self._index_center_x]) < 1e-14)
        assert (np.abs(_y[self._index_center_y]) < 1e-14)

        self._dx = _x[1] - _x[0]
        self._dy = _y[1] - _y[0]

        self._Lx = self._Jx * self._dx
        self._Ly = self._Jy * self._dy

        self._x = torch.tensor(_x, dtype=torch.float64, device=self._device)
        self._y = torch.tensor(_y, dtype=torch.float64, device=self._device)

        self._x_2d = torch.reshape(self._x, (self._Jx, 1))
        self._y_2d = torch.reshape(self._y, (1, self._Jy))

    def init_potential(self, potential, params_user):

        params_solver = {
            "x_2d": self._x_2d,
            "y_2d": self._y_2d,
            "Lx": self._Lx,
            "Ly": self._Ly,
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "m_atom": self._m_atom,
            "unit_length": self._units.unit_length,
            "unit_time": self._units.unit_time,
            "unit_mass": self._units.unit_mass,
            "unit_energy": self._units.unit_energy,
            "unit_frequency": self._units.unit_frequency,
            "device": self._device
        }

        self.potential = potential(params_solver, params_user)

    def set_external_potential(self, *, t, u):

        _t = t / self._units.unit_time

        self._V = self.potential.eval(u)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau = True, return_residuals = False):

        _tau = tau / self._units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core_gpe_2d.compute_ground_state_solution(
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

    def set_u_of_times(self, u_of_times):

        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):

        n_start = kwargs["n_start"]
        n_inc = kwargs["n_inc"]

        mue_shift = kwargs["mue_shift"] / self._units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            if self.u_of_times.ndim > 1:

                u = 0.5 * (self.u_of_times[:, n] + self.u_of_times[:, n + 1])

            else:

                u = 0.5 * (self.u_of_times[n] + self.u_of_times[n + 1])

            self._V = self.potential.eval(u)

            self._psi = qsolve_core_gpe_2d.propagate_gpe(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                self.dt,
                mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def init_time_evolution(self, **kwargs):

        self.t_final = kwargs["t_final"] / self._units.unit_time
        self.dt = kwargs["dt"] / self._units.unit_time

        self.n_time_steps = int(np.round(self.t_final / self.dt))

        self.n_times = self.n_time_steps + 1

        assert (np.abs(self.n_time_steps * self.dt - self.t_final)) < 1e-14

        self._times = self.dt * np.arange(self.n_times)

        assert (np.abs(self._times[-1] - self.t_final)) < 1e-14

    def compute_n_atoms(self, identifier):

        if identifier == "psi":

            n_atoms = qsolve_core_gpe_2d.compute_n_atoms(self._psi, self._dx, self._dy)

        else:

            message = 'identifier \'{0:s}\' not supported for this operation'.format(identifier)

            raise Exception(message)

        return n_atoms

    def compute_chemical_potential(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            mue = qsolve_core_gpe_2d.compute_chemical_potential(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                self._hbar,
                self._m_atom,
                self._g)

        else:

            message = 'compute_chemical_potential(self, identifier, **kwargs): ' \
                      'identifier \'{0:s}\'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self._units.unit_energy * mue

        else:

            return mue

    def compute_E_total(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E = qsolve_core_gpe_2d.compute_total_energy(self._psi, self._V, self._dx, self._dy, self._hbar, self._m_atom,
                                                        self._g)

        else:

            message = 'compute_E_total(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self._units.unit_energy * E

        else:

            return E

    def compute_E_kinetic(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_kinetic = qsolve_core_gpe_2d.compute_kinetic_energy(
                self._psi,
                self._dx,
                self._dy,
                self._hbar,
                self._m_atom)

        else:

            message = 'compute_E_kinetic(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self._units.unit_energy * E_kinetic

        else:

            return E_kinetic

    def compute_E_potential(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_potential = qsolve_core_gpe_2d.compute_potential_energy(self._psi, self._V, self._dx, self._dy)

        else:

            message = 'compute_E_potential(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self._units.unit_energy * E_potential

        else:

            return E_potential

    def compute_E_interaction(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_interaction = qsolve_core_gpe_2d.compute_interaction_energy(
                self._psi,
                self._dx,
                self._dy,
                self._g)

        else:

            message = 'compute_E_interaction(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self._units.unit_energy * E_interaction

        else:

            return E_interaction

    @property
    def x(self):
        return self._units.unit_length * self._x.cpu().numpy()

    @property
    def y(self):
        return self._units.unit_length * self._y.cpu().numpy()

    @property
    def times(self):
        return self._units.unit_time * self._times

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

    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)
