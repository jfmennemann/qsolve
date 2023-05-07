import torch

import numpy as np

import sys

import math

from scipy import constants

from qsolve.core import qsolve_core_gpe_1d

from .getter_functions import get

from qsolve.primes import get_prime_factors
from qsolve.units import Units
# from qsolve.parameter import Parameter


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
        hbar_si = constants.hbar
        mu_B_si = constants.physical_constants['Bohr magneton'][0]
        k_B_si = constants.Boltzmann
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.units = Units.solver_units(kwargs['m_atom'], dim=1)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.hbar = hbar_si / self.units.unit_hbar
        self.mu_B = mu_B_si / self.units.unit_bohr_magneton
        self.k_B = k_B_si / self.units.unit_k_B

        self.m_atom = kwargs['m_atom'] / self.units.unit_mass
        self.a_s = kwargs['a_s'] / self.units.unit_length

        self.omega_perp = kwargs['omega_perp'] / self.units.unit_frequency

        g_3d = 4.0 * constants.pi * self.hbar ** 2 * self.a_s / self.m_atom

        a_perp = math.sqrt(self.hbar / (self.m_atom * self.omega_perp))

        self.g = g_3d / (2 * math.pi * a_perp**2)

        assert (self.hbar == 1.0)
        assert (self.mu_B == 1.0)
        assert (self.k_B == 1.0)

        assert (self.m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.x_min = kwargs['x_min'] / self.units.unit_length
        self.x_max = kwargs['x_max'] / self.units.unit_length

        self.Jx = kwargs['Jx']

        prime_factors_Jx = get_prime_factors(self.Jx)

        assert (np.max(prime_factors_Jx) < 11)

        assert (self.Jx % 2 == 0)

        x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)

        self.index_center_x = np.argmin(np.abs(x))

        assert (np.abs(x[self.index_center_x]) < 1e-14)

        self.dx = x[1] - x[0]

        self.Lx = self.Jx * self.dx

        self.x = torch.tensor(x, dtype=torch.float64, device=self.device)
        # -----------------------------------------------------------------------------------------

        self.q = {
            "hbar": self.hbar,
            "mu_B": self.mu_B,
            "k_B": self.k_B,
            "m_atom": self.m_atom
        }

        self.p = None

        self.V = None
        self.calc_V = None

        self.psi_0 = None
        self.psi = None

        self.t_final = None
        self.dt = None
        self.n_time_steps = None
        self.n_times = None
        self.times = None

        self.u_of_times = None

        self.vec_res_ground_state_computation = None
        self.vec_iter_ground_state_computation = None

    def init_potential(self, calc_V, parameters_potential):

        self.calc_V = calc_V

        self.p = {}

        for key, p in parameters_potential.items():

            if p.dimension == 'frequency':
                value_solver = p.value / self.units.unit_frequency
            elif p.dimension == 'mass':
                value_solver = p.value / self.units.unit_mass
            else:
                # message = 'compute_chemical_potential(self, identifier, **kwargs): ' \
                #           'identifier \'{0:s}\'not supported'.format(identifier)
                raise Exception('unknown dimension')

            self.p[key] = value_solver

    def update_parameters_potential(self, parameters_potential):

        self.p = {}

        for key, p in parameters_potential.items():

            if p.dimension == 'frequency':
                value_solver = p.value / self.units.unit_frequency
            elif p.dimension == 'mass':
                value_solver = p.value / self.units.unit_mass
            else:
                # message = 'compute_chemical_potential(self, identifier, **kwargs): ' \
                #           'identifier \'{0:s}\'not supported'.format(identifier)
                raise Exception('unknown dimension')

            self.p[key] = value_solver

    def set_V(self, t=None, u=None):

        if t is not None:

            t = t / self.units.unit_time

        self.V = self.calc_V(self.x, t, u, self.p, self.q)

    def set_psi(self, identifier, **kwargs):

        if identifier == 'numpy':

            array_numpy = kwargs['array']

            self.psi = torch.tensor(array_numpy / self.units.unit_wave_function, device=self.device)

        else:

            error_message = 'set_psi(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            exit(error_message)

    def compute_ground_state_solution(self, **kwargs):

        tau = kwargs["tau"] / self.units.unit_time

        n_iter = kwargs["n_iter"]

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        if "adaptive_tau" in kwargs:

            adaptive_tau = kwargs["adaptive_tau"]

        else:

            adaptive_tau = True

        N = kwargs["N"]

        psi_0, vec_res, vec_iter = qsolve_core_gpe_1d.compute_ground_state_solution(
            self.V,
            self.dx,
            tau,
            adaptive_tau,
            n_iter,
            N,
            self.hbar,
            self.m_atom,
            self.g)

        self.psi_0 = psi_0

        self.vec_res_ground_state_computation = vec_res
        self.vec_iter_ground_state_computation = vec_iter

    def set_u_of_times(self, u_of_times):
        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):

        n_start = kwargs["n_start"]
        n_inc = kwargs["n_inc"]

        mue_shift = kwargs["mue_shift"] / self.units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            t = self.times[n]

            if self.u_of_times.ndim > 1:

                u = 0.5 * (self.u_of_times[:, n] + self.u_of_times[:, n + 1])

            else:

                u = 0.5 * (self.u_of_times[n] + self.u_of_times[n + 1])

            self.V = self.calc_V(self.x, t, u, self.p, self.q)

            self.psi = qsolve_core_gpe_1d.propagate_gpe(
                self.psi,
                self.V,
                self.dx,
                self.dt,
                mue_shift,
                self.hbar,
                self.m_atom,
                self.g)

            n_local = n_local + 1

    def init_time_evolution(self, **kwargs):

        self.t_final = kwargs["t_final"] / self.units.unit_time
        self.dt = kwargs["dt"] / self.units.unit_time

        self.n_time_steps = int(np.round(self.t_final / self.dt))

        self.n_times = self.n_time_steps + 1

        assert (np.abs(self.n_time_steps * self.dt - self.t_final)) < 1e-14

        self.times = self.dt * np.arange(self.n_times)

        assert (np.abs(self.times[-1] - self.t_final)) < 1e-14

    def compute_n_atoms(self, identifier):
        if identifier == "psi":
            n_atoms = qsolve_core_gpe_1d.compute_n_atoms(self.psi, self.dx)
        elif identifier == "psi_0":
            n_atoms = qsolve_core_gpe_1d.compute_n_atoms(self.psi_0, self.dx)
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

            mue = qsolve_core_gpe_1d.compute_chemical_potential(
                self.psi,
                self.V,
                self.dx,
                self.hbar,
                self.m_atom,
                self.g)

        elif identifier == "psi_0":

            mue = qsolve_core_gpe_1d.compute_chemical_potential(
                self.psi_0,
                self.V,
                self.dx,
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

    def compute_E_total(self, identifier):

        if identifier == "psi":
            psi_tmp = self.psi
        elif identifier == "psi_0":
            psi_tmp = self.psi_0
        else:
            message = 'compute_E_total(self, identifier): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)
            raise Exception(message)

        E = qsolve_core_gpe_1d.compute_total_energy(psi_tmp, self.V, self.dx, self.hbar, self.m_atom, self.g)

        return self.units.unit_energy * E

    def compute_E_kinetic(self, identifier):

        if identifier == "psi":
            psi_tmp = self.psi
        elif identifier == "psi_0":
            psi_tmp = self.psi_0
        else:
            message = 'compute_E_kinetic(self, identifier): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)
            raise Exception(message)

        E_kinetic = qsolve_core_gpe_1d.compute_kinetic_energy(psi_tmp, self.dx, self.hbar, self.m_atom)

        return self.units.unit_energy * E_kinetic

    def compute_E_potential(self, identifier):

        if identifier == "psi":
            psi_tmp = self.psi
        elif identifier == "psi_0":
            psi_tmp = self.psi_0
        else:
            message = 'compute_E_potential(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)
            raise Exception(message)

        E_potential = qsolve_core_gpe_1d.compute_potential_energy(psi_tmp, self.V, self.dx)

        return self.units.unit_energy * E_potential

    def compute_E_interaction(self, identifier):

        if identifier == "psi":
            psi_tmp = self.psi
        elif identifier == "psi_0":
            psi_tmp = self.psi_0
        else:
            message = 'compute_E_interaction(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)
            raise Exception(message)

        E_interaction = qsolve_core_gpe_1d.compute_interaction_energy(psi_tmp, self.dx, self.g)

        return self.units.unit_energy * E_interaction


    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)

    @property
    def _x(self):
        return self.units.unit_length * self.x.cpu().numpy()

    def get(self, identifier, **kwargs):
        return get(self, identifier, kwargs)