import torch

import numpy as np

import sys

import math

from scipy import constants

from qsolve.core import qsolve_core_gpe_1d

from .set_psi import set_psi

from .getter_functions import get

from .n_atoms import compute_n_atoms

from .energies import compute_E_total
from .energies import compute_E_kinetic
from .energies import compute_E_potential
from .energies import compute_E_interaction

from .chemical_potential import compute_chemical_potential

from .compute_ground_state_solution import compute_ground_state_solution

from .init_time_evolution import init_time_evolution

from qsolve.utils.primes import get_prime_factors


class SolverGPE1D(object):

    def __init__(self, **kwargs):

        self.eval_V = kwargs['eval_V']

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
        self.units = kwargs['units']
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

        self.V = None
        self.psi = None

        self.u_of_times = None

    def set_V(self, t, u, p):
        self.V = self.eval_V(self.x, t, u, p)

    def set_psi(self, identifier, **kwargs):
        set_psi(self, identifier, kwargs)

    def compute_ground_state_solution(self, **kwargs):
        compute_ground_state_solution(self, kwargs)

    def set_u_of_times(self, u_of_times):
        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):
        qsolve_core_gpe_1d.propagate_gpe(self, kwargs)

    def init_time_evolution(self, **kwargs):
        init_time_evolution(self, kwargs)

    @property
    def name(self):
        return ...

    def get(self, identifier, **kwargs):
        return get(self, identifier, kwargs)

    def compute_n_atoms(self, identifier):
        return compute_n_atoms(self, identifier)

    def compute_chemical_potential(self, identifier, **kwargs):
        return compute_chemical_potential(self, identifier, kwargs)

    def compute_E_total(self, identifier, **kwargs):
        return compute_E_total(self, identifier, kwargs)

    def compute_E_kinetic(self, identifier, **kwargs):
        return compute_E_kinetic(self, identifier, kwargs)

    def compute_E_potential(self, identifier, **kwargs):
        return compute_E_potential(self, identifier, kwargs)

    def compute_E_interaction(self, identifier, **kwargs):
        return compute_E_interaction(self, identifier, kwargs)

    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)
