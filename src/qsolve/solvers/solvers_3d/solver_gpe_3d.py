import sys
import torch

import scipy

import numpy as np

from qsolve.primes import get_prime_factors

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
        self._hbar = scipy.constants.hbar / self._units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self._m_atom = kwargs['m_atom'] / self._units.unit_mass
        self._a_s = kwargs['a_s'] / self._units.unit_length
        self._g = 4.0 * scipy.constants.pi * self._hbar ** 2 * self._a_s / self._m_atom

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # ---------------------------------------------------------------------------------------------

    def init_grid(self, **kwargs):

        self.x_min = kwargs['x_min'] / self._units.unit_length
        self.x_max = kwargs['x_max'] / self._units.unit_length

        self.y_min = kwargs['y_min'] / self._units.unit_length
        self.y_max = kwargs['y_max'] / self._units.unit_length

        self.z_min = kwargs['z_min'] / self._units.unit_length
        self.z_max = kwargs['z_max'] / self._units.unit_length

        self.Jx = kwargs['Jx']
        self.Jy = kwargs['Jy']
        self.Jz = kwargs['Jz']

        prime_factors_Jx = get_prime_factors(self.Jx)
        prime_factors_Jy = get_prime_factors(self.Jy)
        prime_factors_Jz = get_prime_factors(self.Jz)

        assert (np.max(prime_factors_Jx) < 11)
        assert (np.max(prime_factors_Jy) < 11)
        assert (np.max(prime_factors_Jz) < 11)

        assert (self.Jx % 2 == 0)
        assert (self.Jy % 2 == 0)
        assert (self.Jz % 2 == 0)

        x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)
        y = np.linspace(self.y_min, self.y_max, self.Jy, endpoint=False)
        z = np.linspace(self.z_min, self.z_max, self.Jz, endpoint=False)

        self.index_center_x = np.argmin(np.abs(x))
        self.index_center_y = np.argmin(np.abs(y))
        self.index_center_z = np.argmin(np.abs(z))

        assert (np.abs(x[self.index_center_x]) < 1e-14)
        assert (np.abs(y[self.index_center_y]) < 1e-14)
        assert (np.abs(z[self.index_center_z]) < 1e-14)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self.Lx = self.Jx * self.dx
        self.Ly = self.Jy * self.dy
        self.Lz = self.Jz * self.dz

        self.x = torch.tensor(x, dtype=torch.float64, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float64, device=self.device)
        self.z = torch.tensor(z, dtype=torch.float64, device=self.device)

        self.x_3d = torch.reshape(self.x, (self.Jx, 1, 1))
        self.y_3d = torch.reshape(self.y, (1, self.Jy, 1))
        self.z_3d = torch.reshape(self.z, (1, 1, self.Jz))

    def init_potential(self, Potential, params):

        params_solver = {
            "x_3d": self.x_3d,
            "y_3d": self.y_3d,
            "z_3d": self.z_3d,
            "Lx": self.Lx,
            "Ly": self.Ly,
            "Lz": self.Lz,
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "m_atom": self._m_atom,
            "unit_length": self._units.unit_length,
            "unit_time": self._units.unit_time,
            "unit_mass": self._units.unit_mass,
            "unit_energy": self._units.unit_energy,
            "unit_frequency": self._units.unit_frequency,
            "device": self.device
        }

        self.potential = Potential(params_solver, params)

    def set_V(self, **kwargs):

        u = kwargs['u']

        self.V = self.potential.eval(u)

    def set_psi(self, identifier, **kwargs):

        if identifier == 'numpy':

            array_numpy = kwargs['array']

            self.psi = torch.tensor(array_numpy / self._units.unit_wave_function, device=self.device)

        else:

            error_message = 'set_psi(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            exit(error_message)

    def compute_ground_state_solution(self, **kwargs):

        tau = kwargs["tau"] / self._units.unit_time

        n_iter = kwargs["n_iter"]

        if n_iter < 2500:
            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        if "adaptive_tau" in kwargs:

            adaptive_tau = kwargs["adaptive_tau"]

        else:

            adaptive_tau = True

        N = kwargs["N"]

        psi_0, vec_res, vec_iter = qsolve_core_gpe_3d.compute_ground_state_solution(
            self.V,
            self.dx,
            self.dy,
            self.dz,
            tau,
            adaptive_tau,
            n_iter,
            N,
            self._hbar,
            self._m_atom,
            self._g)

        self.psi_0 = psi_0

        self.vec_res_ground_state_computation = vec_res
        self.vec_iter_ground_state_computation = vec_iter

    def init_sgpe_z_eff(self, **kwargs):

        def __compute_filter_z(z, z1, z2, s):
            Jz = z.shape[0]

            filter_z_1st = 1.0 / (1.0 + torch.exp(-(z - z1) / s))
            filter_z_2nd = 1.0 / (1.0 + torch.exp((z - z2) / s))

            filter_z = filter_z_1st + filter_z_2nd - 1.0

            filter_z = torch.reshape(filter_z, (1, 1, Jz))

            return filter_z

        self.T_des_sgpe = kwargs["T_temp_des"] / self._units.unit_temperature
        self.mue_des_sgpe = kwargs["mue_des"] / self._units.unit_energy
        self.gamma_sgpe = kwargs["gamma"]
        self.dt_sgpe = kwargs["dt"] / self._units.unit_time

        z1 = kwargs["filter_z1"] / self._units.unit_length
        z2 = kwargs["filter_z2"] / self._units.unit_length

        s = kwargs["filter_z_s"] / self._units.unit_length

        self.filter_z_sgpe = __compute_filter_z(self.z, z1, z2, s)

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

            self.V = self.potential.eval(u)

            self.psi = qsolve_core_gpe_3d.propagate_gpe(
                self.psi,
                self.V,
                self.dx,
                self.dy,
                self.dz,
                self.dt,
                mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def init_time_of_flight(self, params):
        qsolve_core_gpe_3d.init_time_of_flight(self, params)

    def compute_time_of_flight(self, **kwargs):
        qsolve_core_gpe_3d.compute_time_of_flight(self, kwargs)

    def propagate_sgpe_z_eff(self, **kwargs):
        qsolve_core_gpe_3d.propagate_sgpe_z_eff(self, kwargs)

    def init_time_evolution(self, **kwargs):

        self.t_final = kwargs["t_final"] / self._units.unit_time
        self.dt = kwargs["dt"] / self._units.unit_time

        self.n_time_steps = int(np.round(self.t_final / self.dt))

        self.n_times = self.n_time_steps + 1

        assert (np.abs(self.n_time_steps * self.dt - self.t_final)) < 1e-14

        self.times = self.dt * np.arange(self.n_times)

        assert (np.abs(self.times[-1] - self.t_final)) < 1e-14

    def get(self, identifier, **kwargs):

        units = "si_units"

        if identifier == "Jx":

            return self.Jx

        elif identifier == "Jy":

            return self.Jy

        elif identifier == "Jz":

            return self.Jz

        elif identifier == "index_center_x":

            return self.index_center_x

        elif identifier == "index_center_y":

            return self.index_center_y

        elif identifier == "index_center_z":

            return self.index_center_z

        elif identifier == "x":

            x = self.x.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * x

            else:

                return x

        elif identifier == "y":

            y = self.y.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * y

            else:

                return y

        elif identifier == "z":

            z = self.z.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * z

            else:

                return z

        elif identifier == "dx":

            if units == "si_units":

                return self._units.unit_length * self.dx

            else:

                return self.dx

        elif identifier == "dy":

            if units == "si_units":

                return self._units.unit_length * self.dy

            else:

                return self.dy

        elif identifier == "dz":

            if units == "si_units":

                return self._units.unit_length * self.dz

            else:

                return self.dz

        elif identifier == "Lx":

            if units == "si_units":

                return self._units.unit_length * self.Lx

            else:

                return self.Lx

        elif identifier == "Ly":

            if units == "si_units":

                return self._units.unit_length * self.Ly

            else:

                return self.Ly

        elif identifier == "Lz":

            if units == "si_units":

                return self._units.unit_length * self.Lz

            else:

                return self.Lz

        elif identifier == "times":

            if units == "si_units":

                return self._units.unit_time * self.times

            else:

                return self.times

        elif identifier == "V":

            V = self.V.cpu().numpy()

            if units == "si_units":

                return self._units.unit_energy * V

            else:

                return V

        elif identifier == "psi_0":

            psi_0 = self.psi_0.cpu().numpy()

            if units == "si_units":

                return self._units.unit_wave_function * psi_0

            else:

                return psi_0

        elif identifier == "psi":

            psi = self.psi.cpu().numpy()

            if units == "si_units":

                return self._units.unit_wave_function * psi

            else:

                return psi

        elif identifier == "filter_z_sgpe":

            filter_z_sgpe = self.filter_z_sgpe.cpu().numpy()

            filter_z_sgpe = np.squeeze(filter_z_sgpe)

            return filter_z_sgpe

        elif identifier == "psi_tof_free_gpe":

            psi_tof_free_gpe = self.psi_tof_free_gpe.cpu().numpy()

            if units == "si_units":

                return self._units.unit_wave_function * psi_tof_free_gpe

            else:

                return psi_tof_free_gpe

        elif identifier == "psi_f_tof_free_schroedinger":

            psi_f_tof_free_schroedinger = self.psi_f_tof_free_schroedinger.cpu().numpy()

            if units == "si_units":

                return self._units.unit_wave_function * psi_f_tof_free_schroedinger

            else:

                return psi_f_tof_free_schroedinger

        elif identifier == "x_tof_free_gpe":

            x_tof_free_gpe = self.x_tof_free_gpe.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * x_tof_free_gpe

            else:

                return x_tof_free_gpe

        elif identifier == "y_tof_free_gpe":

            y_tof_free_gpe = self.y_tof_free_gpe.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * y_tof_free_gpe

            else:

                return y_tof_free_gpe

        elif identifier == "z_tof_free_gpe":

            z_tof_free_gpe = self.z_tof_free_gpe.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * z_tof_free_gpe

            else:

                return z_tof_free_gpe

        elif identifier == "x_tof_final":

            x_f_tof_free_schroedinger = self.x_f_tof_free_schroedinger.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * x_f_tof_free_schroedinger

            else:

                return x_f_tof_free_schroedinger

        elif identifier == "y_tof_final":

            y_f_tof_free_schroedinger = self.y_f_tof_free_schroedinger.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * y_f_tof_free_schroedinger

            else:

                return y_f_tof_free_schroedinger

        elif identifier == "z_tof_final":

            z_f_tof_free_schroedinger = self.z_f_tof_free_schroedinger.cpu().numpy()

            if units == "si_units":

                return self._units.unit_length * z_f_tof_free_schroedinger

            else:

                return z_f_tof_free_schroedinger

        elif identifier == "vec_res_ground_state_computation":

            return self.vec_res_ground_state_computation.cpu().numpy()

        elif identifier == "vec_iter_ground_state_computation":

            return self.vec_iter_ground_state_computation.cpu().numpy()

        else:

            message = 'get(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            raise Exception(message)

    def compute_n_atoms(self, identifier):

        if identifier == "psi":

            n_atoms = qsolve_core_gpe_3d.compute_n_atoms(self.psi, self.dx, self.dy, self.dz)

        elif identifier == "psi_0":

            n_atoms = qsolve_core_gpe_3d.compute_n_atoms(self.psi_0, self.dx, self.dy, self.dz)

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

            mue = qsolve_core_gpe_3d.compute_chemical_potential(
                self.psi,
                self.V,
                self.dx,
                self.dy,
                self.dz,
                self._hbar,
                self._m_atom,
                self._g)

        elif identifier == "psi_0":

            mue = qsolve_core_gpe_3d.compute_chemical_potential(
                self.psi_0,
                self.V,
                self.dx,
                self.dy,
                self.dz,
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

            E = qsolve_core_gpe_3d.compute_total_energy(self.psi, self.V, self.dx, self.dy, self.dz, self._hbar,
                                                        self._m_atom, self._g)

        elif identifier == "psi_0":

            E = qsolve_core_gpe_3d.compute_total_energy(self.psi_0, self.V, self.dx, self.dy, self.dz, self._hbar,
                                                        self._m_atom, self._g)

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

            E_kinetic = qsolve_core_gpe_3d.compute_kinetic_energy(
                self.psi,
                self.dx,
                self.dy,
                self.dz,
                self._hbar,
                self._m_atom)

        elif identifier == "psi_0":

            E_kinetic = qsolve_core_gpe_3d.compute_kinetic_energy(
                self.psi_0,
                self.dx,
                self.dy,
                self.dz,
                self._hbar,
                self._m_atom)

        elif identifier == "psi_tof_free_gpe":

            E_kinetic = qsolve_core_gpe_3d.compute_kinetic_energy(
                self.psi_tof_free_gpe,
                self.dx_tof_free_gpe,
                self.dy_tof_free_gpe,
                self.dz_tof_free_gpe,
                self._hbar,
                self._m_atom)

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

            E_potential = qsolve_core_gpe_3d.compute_potential_energy(self.psi, self.V, self.dx, self.dy, self.dz)

        elif identifier == "psi_0":

            E_potential = qsolve_core_gpe_3d.compute_potential_energy(self.psi_0, self.V, self.dx, self.dy, self.dz)

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

            E_interaction = qsolve_core_gpe_3d.compute_interaction_energy(
                self.psi,
                self.dx,
                self.dy,
                self.dz,
                self._g)

        elif identifier == "psi_0":

            E_interaction = qsolve_core_gpe_3d.compute_interaction_energy(
                self.psi_0,
                self.dx,
                self.dy,
                self.dz,
                self._g)

        elif identifier == "psi_tof_free_gpe":

            E_interaction = qsolve_core_gpe_3d.compute_interaction_energy(
                self.psi_tof_free_gpe,
                self.dx_tof_free_gpe,
                self.dy_tof_free_gpe,
                self.dz_tof_free_gpe,
                self._g)

        else:

            message = 'compute_E_interaction(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        if units == "si_units":

            return self.units.unit_energy * E_interaction

        else:

            return E_interaction

    def compute_density_xy(self, identifier, **kwargs):

        if "rescaling" in kwargs:

            rescaling = kwargs["rescaling"]

        else:

            rescaling = False

        if identifier == "psi_tof_gpe":

            if "index_z" in kwargs:

                index_z = kwargs["index_z"]

            else:

                index_z = self.index_center_z_tof_free_gpe

            density_xy = qsolve_core_gpe_3d.compute_density_xy(self.psi_tof_free_gpe, index_z, rescaling)

        elif identifier == "psi_f_tof_free_schroedinger":

            if "index_z" in kwargs:

                index_z = kwargs["index_z"]

            else:

                index_z = self.index_center_z_f_tof_free_schroedinger

            density_xy = qsolve_core_gpe_3d.compute_density_xy(self.psi_f_tof_free_schroedinger, index_z, rescaling)

        else:

            message = 'compute_density_xz(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            raise Exception(message)

        return density_xy.cpu().numpy()

    def compute_density_xz(self, identifier, **kwargs):

        if "rescaling" in kwargs:

            rescaling = kwargs["rescaling"]

        else:

            rescaling = False

        if identifier == "psi_tof_gpe":

            if "index_y" in kwargs:

                index_y = kwargs["index_y"]

            else:

                index_y = self.index_center_y_tof_free_gpe

            density_xz = qsolve_core_gpe_3d.compute_density_xz(self.psi_tof_free_gpe, index_y, rescaling)

        elif identifier == "psi_f_tof_free_schroedinger":

            if "index_y" in kwargs:

                index_y = kwargs["index_y"]

            else:

                index_y = self.index_center_y_f_tof_free_schroedinger

            density_xz = qsolve_core_gpe_3d.compute_density_xz(self.psi_f_tof_free_schroedinger, index_y, rescaling)

        else:

            message = 'compute_density_xz(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            raise Exception(message)

        return density_xz.cpu().numpy()

    def compute_spectrum_abs_xy(self, identifier, **kwargs):

        if "rescaling" in kwargs:

            rescaling = kwargs["rescaling"]

        else:

            rescaling = False

        if identifier == "psi_tof_gpe":

            if "index_z" in kwargs:

                index_z = kwargs["index_z"]

            else:

                index_z = self.index_center_z_tof_free_gpe

            spectrum_abs_xy = qsolve_core_gpe_3d.compute_spectrum_abs_xy(self.psi_tof_free_gpe, index_z, rescaling)

        else:

            message = 'compute_spectrum_abs_xy(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        spectrum_abs_xy = spectrum_abs_xy.cpu().numpy()

        return spectrum_abs_xy

    def compute_spectrum_abs_xz(self, identifier, **kwargs):

        if "rescaling" in kwargs:

            rescaling = kwargs["rescaling"]

        else:

            rescaling = False

        if identifier == "psi_tof_gpe":

            if "index_y" in kwargs:

                index_y = kwargs["index_y"]

            else:

                index_y = self.index_center_y_tof_free_gpe

            spectrum_abs_xz = qsolve_core_gpe_3d.compute_spectrum_abs_xz(self.psi_tof_free_gpe, index_y, rescaling)

        else:

            message = 'compute_spectrum_abs_xy(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                      'not supported'.format(identifier)

            raise Exception(message)

        spectrum_abs_xz = spectrum_abs_xz.cpu().numpy()

        return spectrum_abs_xz
