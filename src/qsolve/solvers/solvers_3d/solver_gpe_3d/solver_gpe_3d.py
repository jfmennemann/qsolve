import sys
import torch

import scipy

from .init_grid_3d import init_grid

from .init_potential import init_potential

from .set_psi import set_psi
from .set_V import set_V

from .getter_functions import get

from . import densities

from . import spectrum
from . import n_atoms
from . import energies
from . import chemical_potential

from .compute_ground_state_solution import compute_ground_state_solution

from .init_time_evolution import init_time_evolution

from qsolve.core import qsolve_core_gpe_3d

from qsolve.units import Units


class SolverGPE3D(object):

    def __init__(self, **kwargs):

        # -------------------------------------------------------------------------------------------------
        print("Python version:")
        print(sys.version)
        print()
        print("PyTorch version:")
        print(torch.__version__)
        print()
        # -------------------------------------------------------------------------------------------------

        if 'seed' in kwargs:

            seed = kwargs['seed']

        else:

            seed = 0

        torch.manual_seed(seed)

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

        self._units = Units.solver_units(kwargs['m_atom'], dim=3)

        # ---------------------------------------------------------------------------------------------
        self.hbar = scipy.constants.hbar / self._units.unit_hbar
        self.mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self.k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self.m_atom = kwargs['m_atom'] / self._units.unit_mass
        self.a_s = kwargs['a_s'] / self._units.unit_length
        self.g = 4.0 * scipy.constants.pi * self.hbar ** 2 * self.a_s / self.m_atom

        assert (self.hbar == 1.0)
        assert (self.mu_B == 1.0)
        assert (self.k_B == 1.0)

        assert (self.m_atom == 1.0)
        # ---------------------------------------------------------------------------------------------

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

    def init_sgpe_z_eff(self, **kwargs):
        qsolve_core_gpe_3d.init_sgpe_z_eff(self, kwargs)

    def set_u_of_times(self, u_of_times):
        self.u_of_times = u_of_times

    def propagate_gpe(self, **kwargs):
        qsolve_core_gpe_3d.propagate_gpe(self, kwargs)

    def init_time_of_flight(self, params):
        qsolve_core_gpe_3d.init_time_of_flight(self, params)

    def compute_time_of_flight(self, **kwargs):
        qsolve_core_gpe_3d.compute_time_of_flight(self, kwargs)

    def propagate_sgpe_z_eff(self, **kwargs):
        qsolve_core_gpe_3d.propagate_sgpe_z_eff(self, kwargs)

    def init_time_evolution(self, **kwargs):
        init_time_evolution(self, kwargs)

    def get(self, identifier, **kwargs):
        return get(self, identifier, kwargs)

    def compute_n_atoms(self, identifier):
        return n_atoms.compute_n_atoms(self, identifier)

    def compute_chemical_potential(self, identifier, **kwargs):
        return chemical_potential.compute_chemical_potential(self, identifier, kwargs)

    def compute_E_total(self, identifier, **kwargs):
        return energies.compute_E_total(self, identifier, kwargs)

    def compute_E_kinetic(self, identifier, **kwargs):
        return energies.compute_E_kinetic(self, identifier, kwargs)

    def compute_E_potential(self, identifier, **kwargs):
        return energies.compute_E_potential(self, identifier, kwargs)

    def compute_E_interaction(self, identifier, **kwargs):
        return energies.compute_E_interaction(self, identifier, kwargs)

    def compute_density_xy(self, identifier, **kwargs):
        return densities.compute_density_xy(self, identifier, kwargs)

    def compute_density_xz(self, identifier, **kwargs):
        return densities.compute_density_xz(self, identifier, kwargs)

    def compute_spectrum_abs_xy(self, identifier, **kwargs):
        return spectrum.compute_spectrum_abs_xy(self, identifier, kwargs)

    def compute_spectrum_abs_xz(self, identifier, **kwargs):
        return spectrum.compute_spectrum_abs_xz(self, identifier, kwargs)
