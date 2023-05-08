import torch

import scipy

import numpy as np

import sys

import math

from qsolve.core import qsolve_core_gpe_1d

from qsolve.primes import get_prime_factors
from qsolve.units import Units


class SolverGPE1D(object):

    def __init__(self, **kwargs):

        # -----------------------------------------------------------------------------------------
        print("Python version:")
        print(sys.version)
        print()
        print("PyTorch version:")
        print(torch.__version__)
        print()
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = 0

        torch.manual_seed(seed)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        if 'device' in kwargs:

            if kwargs['device'] == 'cuda:0':

                self.device = torch.device('cuda:0')

            elif kwargs['device'] == 'cpu':

                self.device = torch.device('cpu')

            else:

                message = 'device \'{0:s}\' not supported'.format(kwargs['device'])

                raise Exception(message)

        else:

            self.device = torch.device('cpu')

        if 'num_threads_cpu' in kwargs:

            torch.set_num_threads(kwargs['num_threads_cpu'])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.units = Units.solver_units(kwargs['m_atom'], dim=1)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self._hbar = scipy.constants.hbar / self.units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self.units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self.units.unit_k_B

        self._m_atom = kwargs['m_atom'] / self.units.unit_mass
        self._a_s = kwargs['a_s'] / self.units.unit_length

        _omega_perp = kwargs['omega_perp'] / self.units.unit_frequency

        _g_3d = 4.0 * scipy.constants.pi * self._hbar ** 2 * self._a_s / self._m_atom

        _a_perp = math.sqrt(self._hbar / (self._m_atom * _omega_perp))

        self._g = _g_3d / (2 * math.pi * _a_perp**2)

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self._x_min = kwargs['x_min'] / self.units.unit_length
        self._x_max = kwargs['x_max'] / self.units.unit_length

        self._Jx = kwargs['Jx']

        assert (np.max(get_prime_factors(self._Jx)) < 11)

        assert (self._Jx % 2 == 0)

        _x = np.linspace(self._x_min, self._x_max, self._Jx, endpoint=False)

        self._dx = _x[1] - _x[0]

        self._Lx = self._Jx * self._dx

        self._x = torch.tensor(_x, dtype=torch.float64, device=self.device)
        # -----------------------------------------------------------------------------------------

        self._V = None
        self.calc_V = None

        self._psi = None

        self._t_final = None
        self._dt = None
        self._n_time_steps = None
        self._n_times = None
        self._times = None

        self._u_of_times = None

        self._vec_res_ground_state_computation = None
        self._vec_iter_ground_state_computation = None

        self.q = {
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "k_B": self._k_B,
            "m_atom": self._m_atom
        }

        self.p = None

    def init_potential(self, calc_V, parameters_potential):

        self.calc_V = calc_V

        self.p = {}

        for key, p in parameters_potential.items():

            value = p[0]
            unit = p[1]

            if unit == 'm':
                _value = value / self.units.unit_length
            elif unit == 's':
                _value = value / self.units.unit_time
            elif unit == 'Hz':
                _value = value / self.units.unit_frequency
            else:
                raise Exception('unknown unit')

            self.p[key] = _value


    def set_V(self, t=None, u=None):

        if t is not None:

            t = t / self.units.unit_time

        self._V = self.calc_V(self._x, t, u, self.p, self.q)

    def compute_ground_state_solution(self, **kwargs):

        _tau = kwargs["tau"] / self.units.unit_time

        n_iter = kwargs["n_iter"]

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        if "adaptive_tau" in kwargs:

            adaptive_tau = kwargs["adaptive_tau"]

        else:

            adaptive_tau = True

        N = kwargs["n_atoms"]

        _psi_0, vec_res, vec_iter = qsolve_core_gpe_1d.compute_ground_state_solution(
            self._V,
            self._dx,
            _tau,
            adaptive_tau,
            n_iter,
            N,
            self._hbar,
            self._m_atom,
            self._g)

        self._vec_res_ground_state_computation = vec_res
        self._vec_iter_ground_state_computation = vec_iter

        return self.units.unit_wave_function * _psi_0.cpu().numpy()

    def set_u_of_times(self, u_of_times):
        self._u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):

        n_start = kwargs["n_start"]
        n_inc = kwargs["n_inc"]

        _mue_shift = kwargs["mue_shift"] / self.units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            t = self._times[n]

            if self._u_of_times.ndim > 1:

                _u = 0.5 * (self._u_of_times[:, n] + self._u_of_times[:, n + 1])

            else:

                _u = 0.5 * (self._u_of_times[n] + self._u_of_times[n + 1])

            self._V = self.calc_V(self._x, t, _u, self.p, self.q)

            self._psi = qsolve_core_gpe_1d.propagate_gpe(
                self._psi,
                self._V,
                self._dx,
                self._dt,
                _mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def init_time_evolution(self, t_final, dt):

        self._t_final = t_final / self.units.unit_time
        self._dt = dt / self.units.unit_time

        self._n_time_steps = int(np.round(self._t_final / self._dt))

        self._n_times = self._n_time_steps + 1

        assert (np.abs(self._n_time_steps * self._dt - self._t_final)) < 1e-14

        self._times = self._dt * np.arange(self._n_times)

        assert (np.abs(self._times[-1] - self._t_final)) < 1e-14

    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)

    @property
    def x(self):
        return self.units.unit_length * self._x.cpu().numpy()

    @property
    def dx(self):
        return self.units.unit_length * self._dx

    @property
    def times(self):
        return self.units.unit_time * self._times

    @property
    def psi(self):
        return self.units.unit_wave_function * self._psi.cpu().numpy()

    @psi.setter
    def psi(self, value):
        self._psi = torch.tensor(value / self.units.unit_wave_function, device=self.device)

    @property
    def V(self):
        return self.units.unit_energy * self._V.cpu().numpy()

    @property
    def vec_res_ground_state_computation(self):
        return self._vec_res_ground_state_computation.cpu().numpy()

    @property
    def vec_iter_ground_state_computation(self):
        return self._vec_iter_ground_state_computation.cpu().numpy()

    def compute_n_atoms(self):
        return qsolve_core_gpe_1d.compute_n_atoms(self._psi, self._dx)

    def compute_chemical_potential(self):

        _mue = qsolve_core_gpe_1d.compute_chemical_potential(
            self._psi, self._V, self._dx, self._hbar, self._m_atom, self._g)

        return self.units.unit_energy * _mue

    def compute_total_energy(self):

        _E = qsolve_core_gpe_1d.compute_total_energy(self._psi, self._V, self._dx, self._hbar, self._m_atom, self._g)

        return self.units.unit_energy * _E

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core_gpe_1d.compute_kinetic_energy(self._psi, self._dx, self._hbar, self._m_atom)

        return self.units.unit_energy * _E_kinetic

    def compute_potential_energy(self):

        _E_potential = qsolve_core_gpe_1d.compute_potential_energy(self._psi, self._V, self._dx)

        return self.units.unit_energy * _E_potential

    def compute_interaction_energy(self):

        _E_interaction = qsolve_core_gpe_1d.compute_interaction_energy(self._psi, self._dx, self._g)

        return self.units.unit_energy * _E_interaction
