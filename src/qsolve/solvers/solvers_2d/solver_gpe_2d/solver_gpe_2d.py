import torch

import sys

import math

import numpy as np

from scipy import constants

from qsolve.core import qsolve_core_gpe_2d

from .units import Units

from .getter_functions import get

from qsolve.primes import get_prime_factors


class SolverGPE2D(object):

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
        hbar_si = constants.hbar
        mu_B_si = constants.physical_constants['Bohr magneton'][0]
        k_B_si = constants.Boltzmann
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        unit_mass = kwargs['m_atom']
        unit_length = 1e-6
        unit_time = unit_mass * (unit_length * unit_length) / hbar_si

        unit_electric_current = mu_B_si / (unit_length * unit_length)
        unit_temperature = (unit_mass * unit_length * unit_length) / (k_B_si * unit_time * unit_time)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.units = Units(unit_length, unit_time, unit_mass, unit_electric_current, unit_temperature)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.hbar = hbar_si / self.units.unit_hbar
        self.mu_B = mu_B_si / self.units.unit_bohr_magneton
        self.k_B = k_B_si / self.units.unit_k_B

        self.m_atom = kwargs['m_atom'] / self.units.unit_mass
        self.a_s = kwargs['a_s'] / self.units.unit_length

        self.omega_z = kwargs['omega_z'] / self.units.unit_frequency

        g_3d = 4.0 * constants.pi * self.hbar ** 2 * self.a_s / self.m_atom

        a_z = math.sqrt(self.hbar / (self.m_atom * self.omega_z))

        self.g = g_3d / (math.sqrt(2 * math.pi) * a_z)

        assert (self.hbar == 1.0)
        assert (self.mu_B == 1.0)
        assert (self.k_B == 1.0)

        assert (self.m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        self.u_of_times = None

        # self._V = None

    def init_grid(self, **kwargs):

        self.x_min = kwargs['x_min'] / self.units.unit_length
        self.x_max = kwargs['x_max'] / self.units.unit_length

        self.y_min = kwargs['y_min'] / self.units.unit_length
        self.y_max = kwargs['y_max'] / self.units.unit_length

        self.Jx = kwargs['Jx']
        self.Jy = kwargs['Jy']

        prime_factors_Jx = get_prime_factors(self.Jx)
        prime_factors_Jy = get_prime_factors(self.Jy)

        assert (np.max(prime_factors_Jx) < 11)
        assert (np.max(prime_factors_Jy) < 11)

        assert (self.Jx % 2 == 0)
        assert (self.Jy % 2 == 0)

        x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)
        y = np.linspace(self.y_min, self.y_max, self.Jy, endpoint=False)

        self._index_center_x = np.argmin(np.abs(x))
        self._index_center_y = np.argmin(np.abs(y))

        assert (np.abs(x[self._index_center_x]) < 1e-14)
        assert (np.abs(y[self._index_center_y]) < 1e-14)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.Lx = self.Jx * self.dx
        self.Ly = self.Jy * self.dy

        self._x = torch.tensor(x, dtype=torch.float64, device=self.device)
        self._y = torch.tensor(y, dtype=torch.float64, device=self.device)

        self.x_2d = torch.reshape(self._x, (self.Jx, 1))
        self.y_2d = torch.reshape(self._y, (1, self.Jy))

    def init_potential(self, potential, params_user):

        params_solver = {
            "x_2d": self.x_2d,
            "y_2d": self.y_2d,
            "Lx": self.Lx,
            "Ly": self.Ly,
            "hbar": self.hbar,
            "mu_B": self.mu_B,
            "m_atom": self.m_atom,
            "unit_length": self.units.unit_length,
            "unit_time": self.units.unit_time,
            "unit_mass": self.units.unit_mass,
            "unit_energy": self.units.unit_energy,
            "unit_frequency": self.units.unit_frequency,
            "device": self.device
        }

        self.potential = potential(params_solver, params_user)

    def set_V(self, **kwargs):

        u = kwargs['u']

        self._V = self.potential.eval(u)

    def set_psi(self, identifier, **kwargs):

        if identifier == 'numpy':

            array_numpy = kwargs['array']

            self.psi = torch.tensor(array_numpy / self.units.unit_wave_function, device=self.device)

        else:

            error_message = 'set_psi(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            exit(error_message)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau = True, return_residuals = False):

        _tau = tau / self.units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core_gpe_2d.compute_ground_state_solution(
            self._V,
            self.dx,
            self.dy,
            _tau,
            adaptive_tau,
            n_iter,
            n_atoms,
            self.hbar,
            self.m_atom,
            self.g)

        self.psi_0 = _psi_0

        self.vec_res_ground_state_computation = vec_res
        self.vec_iter_ground_state_computation = vec_iter

        if return_residuals:

            return self.units.unit_wave_function * _psi_0.cpu().numpy(), vec_res, vec_iter

        else:

            return self.units.unit_wave_function * _psi_0.cpu().numpy()

    def set_u_of_times(self, u_of_times):

        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):

        n_start = kwargs["n_start"]
        n_inc = kwargs["n_inc"]

        mue_shift = kwargs["mue_shift"] / self.units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            if self.u_of_times.ndim > 1:

                u = 0.5 * (self.u_of_times[:, n] + self.u_of_times[:, n + 1])

            else:

                u = 0.5 * (self.u_of_times[n] + self.u_of_times[n + 1])

            self._V = self.potential.eval(u)

            self.psi = qsolve_core_gpe_2d.propagate_gpe(
                self.psi,
                self._V,
                self.dx,
                self.dy,
                self.dt,
                mue_shift,
                self.hbar,
                self.m_atom,
                self.g)

            n_local = n_local + 1

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)

    def init_time_evolution(self, **kwargs):

        self.t_final = kwargs["t_final"] / self.units.unit_time
        self.dt = kwargs["dt"] / self.units.unit_time

        self.n_time_steps = int(np.round(self.t_final / self.dt))

        self.n_times = self.n_time_steps + 1

        assert (np.abs(self.n_time_steps * self.dt - self.t_final)) < 1e-14

        self._times = self.dt * np.arange(self.n_times)

        assert (np.abs(self._times[-1] - self.t_final)) < 1e-14

    def get(self, identifier, **kwargs):

        return get(self, identifier, kwargs)

    def compute_n_atoms(self, identifier):

        if identifier == "psi":

            n_atoms = qsolve_core_gpe_2d.compute_n_atoms(self.psi, self.dx, self.dy)

        elif identifier == "psi_0":

            n_atoms = qsolve_core_gpe_2d.compute_n_atoms(self.psi_0, self.dx, self.dy)

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
                self.psi,
                self._V,
                self.dx,
                self.dy,
                self.hbar,
                self.m_atom,
                self.g)

        elif identifier == "psi_0":

            mue = qsolve_core_gpe_2d.compute_chemical_potential(
                self.psi_0,
                self._V,
                self.dx,
                self.dy,
                self.hbar,
                self.m_atom,
                self.g)

        else:

            message = 'compute_chemical_potential(self, identifier, **kwargs): ' \
                      'identifier \'{0:s}\'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * mue

        else:

            return mue

    def compute_E_total(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E = qsolve_core_gpe_2d.compute_total_energy(self.psi, self._V, self.dx, self.dy, self.hbar, self.m_atom,
                                                        self.g)

        elif identifier == "psi_0":

            E = qsolve_core_gpe_2d.compute_total_energy(self.psi_0, self._V, self.dx, self.dy, self.hbar, self.m_atom,
                                                        self.g)

        else:

            message = 'compute_E_total(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * E

        else:

            return E

    def compute_E_kinetic(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_kinetic = qsolve_core_gpe_2d.compute_kinetic_energy(
                self.psi,
                self.dx,
                self.dy,
                self.hbar,
                self.m_atom)

        elif identifier == "psi_0":

            E_kinetic = qsolve_core_gpe_2d.compute_kinetic_energy(
                self.psi_0,
                self.dx,
                self.dy,
                self.hbar,
                self.m_atom)

        else:

            message = 'compute_E_kinetic(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * E_kinetic

        else:

            return E_kinetic

    def compute_E_potential(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_potential = qsolve_core_gpe_2d.compute_potential_energy(self.psi, self._V, self.dx, self.dy)

        elif identifier == "psi_0":

            E_potential = qsolve_core_gpe_2d.compute_potential_energy(self.psi_0, self._V, self.dx, self.dy)

        else:

            message = 'compute_E_potential(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * E_potential

        else:

            return E_potential

    def compute_E_interaction(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_interaction = qsolve_core_gpe_2d.compute_interaction_energy(
                self.psi,
                self.dx,
                self.dy,
                self.g)

        elif identifier == "psi_0":

            E_interaction = qsolve_core_gpe_2d.compute_interaction_energy(
                self.psi_0,
                self.dx,
                self.dy,
                self.g)

        else:

            message = 'compute_E_interaction(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * E_interaction

        else:

            return E_interaction

    @property
    def x(self):
        return self.units.unit_length * self._x.cpu().numpy()

    @property
    def y(self):
        return self.units.unit_length * self._y.cpu().numpy()

    @property
    def times(self):
        return self.units.unit_time * self._times

    # if identifier == "index_center_x":
    #
    #     return self.index_center_x
    #
    # elif identifier == "index_center_y":
    #
    #     return self.index_center_y

    @property
    def index_center_x(self):
        return self._index_center_x

    @property
    def index_center_y(self):
        return self._index_center_y

    # if identifier == "V":
    #
    #     V = self._V.cpu().numpy()
    #
    #     if units == "si_units":
    #
    #
    #
    #     else:
    #
    #         return V

    @property
    def V(self):
        return self.units.unit_energy * self._V.cpu().numpy()



    # elif identifier == "vec_res_ground_state_computation":
    #
    #     return self.vec_res_ground_state_computation.cpu().numpy()
    #
    # elif identifier == "vec_iter_ground_state_computation":
    #
    #     return self.vec_iter_ground_state_computation.cpu().numpy()

    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)