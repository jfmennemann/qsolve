import torch

import sys

import math

from scipy import constants

from qsolve.core import qsolve_core_gpe_2d

from .units import Units

from .init_device import init_device

from .init_grid_2d import init_grid

from .init_potential import init_potential

from .set_psi import set_psi
from .set_V import set_V

from .getter_functions import get

from .n_atoms import compute_n_atoms

from .energies import compute_E_total
from .energies import compute_E_kinetic
from .energies import compute_E_potential
from .energies import compute_E_interaction

from .chemical_potential import compute_chemical_potential

from .compute_ground_state_solution import compute_ground_state_solution

from .init_time_evolution import init_time_evolution


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
        init_device(self, kwargs)
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

    def init_grid(self, **kwargs):
        init_grid(self, kwargs)

    def init_potential(self, potential, params):
        init_potential(self, potential, params)

    def set_V(self, **kwargs):
        set_V(self, kwargs)

    def set_psi(self, identifier, **kwargs):
        set_psi(self, identifier, kwargs)

    def compute_ground_state_solution(self, **kwargs):
        compute_ground_state_solution(self, kwargs)

    # def init_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    def set_u_of_times(self, u_of_times):
        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):

        qsolve_core_gpe_2d.propagate_gpe(self, kwargs)

    # def propagate_sgpe_z_eff(self, **kwargs):
    #     qsolve_core_gpe_2d.propagate_sgpe_z_eff(self, kwargs)

    def init_time_evolution(self, **kwargs):
        init_time_evolution(self, kwargs)

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
