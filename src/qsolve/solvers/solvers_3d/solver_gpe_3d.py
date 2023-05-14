import torch

import scipy

import numpy as np

import sys

import math

from qsolve.core import qsolve_core_gpe_3d

from qsolve.primes import get_prime_factors

from qsolve.units import Units


class SolverGPE3D(object):

    def __init__(self, *, m_atom, a_s, seed=0, device='cpu', num_threads_cpu=1):

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

        self._units = Units.solver_units(m_atom, dim=3)

        # ---------------------------------------------------------------------------------------------
        self._hbar = scipy.constants.hbar / self._units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self._m_atom = m_atom / self._units.unit_mass
        self._a_s = a_s / self._units.unit_length
        self._g = 4.0 * math.pi * self._hbar ** 2 * self._a_s / self._m_atom

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # ---------------------------------------------------------------------------------------------

    def init_grid(self, **kwargs):

        self._x_min = kwargs['x_min'] / self._units.unit_length
        self._x_max = kwargs['x_max'] / self._units.unit_length

        self._y_min = kwargs['y_min'] / self._units.unit_length
        self._y_max = kwargs['y_max'] / self._units.unit_length

        self._z_min = kwargs['z_min'] / self._units.unit_length
        self._z_max = kwargs['z_max'] / self._units.unit_length

        self._Jx = kwargs['Jx']
        self._Jy = kwargs['Jy']
        self._Jz = kwargs['Jz']

        assert (np.max(get_prime_factors(self._Jx)) < 11)
        assert (np.max(get_prime_factors(self._Jy)) < 11)
        assert (np.max(get_prime_factors(self._Jz)) < 11)

        assert (self._Jx % 2 == 0)
        assert (self._Jy % 2 == 0)
        assert (self._Jz % 2 == 0)

        _x = np.linspace(self._x_min, self._x_max, self._Jx, endpoint=False)
        _y = np.linspace(self._y_min, self._y_max, self._Jy, endpoint=False)
        _z = np.linspace(self._z_min, self._z_max, self._Jz, endpoint=False)

        self._index_center_x = np.argmin(np.abs(_x))
        self._index_center_y = np.argmin(np.abs(_y))
        self._index_center_z = np.argmin(np.abs(_z))

        assert (np.abs(_x[self._index_center_x]) < 1e-14)
        assert (np.abs(_y[self._index_center_y]) < 1e-14)
        assert (np.abs(_z[self._index_center_z]) < 1e-14)

        self._dx = _x[1] - _x[0]
        self._dy = _y[1] - _y[0]
        self._dz = _z[1] - _z[0]

        self._Lx = self._Jx * self._dx
        self._Ly = self._Jy * self._dy
        self._Lz = self._Jz * self._dz

        self._x = torch.tensor(_x, dtype=torch.float64, device=self._device)
        self._y = torch.tensor(_y, dtype=torch.float64, device=self._device)
        self._z = torch.tensor(_z, dtype=torch.float64, device=self._device)

        self._x_3d = torch.reshape(self._x, (self._Jx, 1, 1))
        self._y_3d = torch.reshape(self._y, (1, self._Jy, 1))
        self._z_3d = torch.reshape(self._z, (1, 1, self._Jz))

    def init_potential(self, Potential, params):

        params_solver = {
            "x_3d": self._x_3d,
            "y_3d": self._y_3d,
            "z_3d": self._z_3d,
            "Lx": self._Lx,
            "Ly": self._Ly,
            "Lz": self._Lz,
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

        self.potential = Potential(params_solver, params)

    def set_V(self, **kwargs):

        u = kwargs['u']

        self._V = self.potential.eval(u)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau=True, return_residuals=False):

        _tau = tau / self._units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core_gpe_3d.compute_ground_state_solution(
            self._V,
            self._dx,
            self._dy,
            self._dz,
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

        self.filter_z_sgpe = __compute_filter_z(self._z, z1, z2, s)

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

            self._psi = qsolve_core_gpe_3d.propagate_gpe(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                self._dz,
                self.dt,
                mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def init_time_of_flight(self, params):

        self.Jx_tof_free_gpe = params["Jx_tof_free_gpe"]
        self.Jy_tof_free_gpe = params["Jy_tof_free_gpe"]
        self.Jz_tof_free_gpe = params["Jz_tof_free_gpe"]

        self.T_tof_total = params["T_tof_total"] / self._units.unit_time
        self.T_tof_free_gpe = params["T_tof_free_gpe"] / self._units.unit_time

        self.T_tof_free_schroedinger = self.T_tof_total - self.T_tof_free_gpe

        self.dt_tof_free_gpe = self.dt

        self.dx_tof_free_gpe = self._dx
        self.dy_tof_free_gpe = self._dy
        self.dz_tof_free_gpe = self._dz

        # ---------------------------------------------------------------------------------------------
        self.n_time_steps_tof_free_gpe = int(np.round(self.T_tof_free_gpe / self.dt_tof_free_gpe))

        assert (self.n_time_steps_tof_free_gpe * self.dt_tof_free_gpe - self.T_tof_free_gpe) < 1e-14
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        assert (self.Jx_tof_free_gpe >= self._Jx)
        assert (self.Jy_tof_free_gpe >= self._Jy)
        assert (self.Jz_tof_free_gpe >= self._Jz)

        assert (self.Jx_tof_free_gpe % 2 == 0)
        assert (self.Jy_tof_free_gpe % 2 == 0)
        assert (self.Jz_tof_free_gpe % 2 == 0)

        prime_factors_Jx_tof_free_gpe = get_prime_factors(self.Jx_tof_free_gpe)
        prime_factors_Jy_tof_free_gpe = get_prime_factors(self.Jy_tof_free_gpe)
        prime_factors_Jz_tof_free_gpe = get_prime_factors(self.Jz_tof_free_gpe)

        assert (np.max(prime_factors_Jx_tof_free_gpe) < 11)
        assert (np.max(prime_factors_Jy_tof_free_gpe) < 11)
        assert (np.max(prime_factors_Jz_tof_free_gpe) < 11)
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        x_tof_free_gpe = self.dx_tof_free_gpe * np.arange(-self.Jx_tof_free_gpe // 2, self.Jx_tof_free_gpe // 2)
        y_tof_free_gpe = self.dy_tof_free_gpe * np.arange(-self.Jy_tof_free_gpe // 2, self.Jy_tof_free_gpe // 2)
        z_tof_free_gpe = self.dz_tof_free_gpe * np.arange(-self.Jz_tof_free_gpe // 2, self.Jz_tof_free_gpe // 2)

        self.index_center_x_tof_free_gpe = np.argmin(np.abs(x_tof_free_gpe))
        self.index_center_y_tof_free_gpe = np.argmin(np.abs(y_tof_free_gpe))
        self.index_center_z_tof_free_gpe = np.argmin(np.abs(z_tof_free_gpe))

        assert (np.abs(x_tof_free_gpe[self.index_center_x_tof_free_gpe]) < 1e-14)
        assert (np.abs(y_tof_free_gpe[self.index_center_y_tof_free_gpe]) < 1e-14)
        assert (np.abs(z_tof_free_gpe[self.index_center_z_tof_free_gpe]) < 1e-14)

        self.x_tof_free_gpe = torch.tensor(x_tof_free_gpe, dtype=torch.float64, device=self._device)
        self.y_tof_free_gpe = torch.tensor(y_tof_free_gpe, dtype=torch.float64, device=self._device)
        self.z_tof_free_gpe = torch.tensor(z_tof_free_gpe, dtype=torch.float64, device=self._device)
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        self.x_0_tof_free_schroedinger = self.x_tof_free_gpe
        self.y_0_tof_free_schroedinger = self.y_tof_free_gpe
        self.z_0_tof_free_schroedinger = self.z_tof_free_gpe

        Jx_tof_final = params["Jx_tof_final"]
        Jy_tof_final = params["Jy_tof_final"]
        Jz_tof_final = params["Jz_tof_final"]

        x_min_tof_final = params["x_min_tof_final"] / self._units.unit_length
        x_max_tof_final = params["x_max_tof_final"] / self._units.unit_length

        y_min_tof_final = params["y_min_tof_final"] / self._units.unit_length
        y_max_tof_final = params["y_max_tof_final"] / self._units.unit_length

        z_min_tof_final = params["z_min_tof_final"] / self._units.unit_length
        z_max_tof_final = params["z_max_tof_final"] / self._units.unit_length

        x_f_tof_free_schroedinger = np.linspace(x_min_tof_final, x_max_tof_final, Jx_tof_final, endpoint=True)
        y_f_tof_free_schroedinger = np.linspace(y_min_tof_final, y_max_tof_final, Jy_tof_final, endpoint=True)
        z_f_tof_free_schroedinger = np.linspace(z_min_tof_final, z_max_tof_final, Jz_tof_final, endpoint=True)

        index_center_x_f_tof_free_schroedinger = np.argmin(np.abs(x_f_tof_free_schroedinger))
        index_center_y_f_tof_free_schroedinger = np.argmin(np.abs(y_f_tof_free_schroedinger))
        index_center_z_f_tof_free_schroedinger = np.argmin(np.abs(z_f_tof_free_schroedinger))

        assert (np.abs(x_f_tof_free_schroedinger[index_center_x_f_tof_free_schroedinger]) < 5e-14)
        assert (np.abs(y_f_tof_free_schroedinger[index_center_y_f_tof_free_schroedinger]) < 5e-14)
        assert (np.abs(z_f_tof_free_schroedinger[index_center_z_f_tof_free_schroedinger]) < 5e-14)

        self.x_f_tof_free_schroedinger = torch.tensor(x_f_tof_free_schroedinger, dtype=torch.float64,
                                                      device=self._device)
        self.y_f_tof_free_schroedinger = torch.tensor(y_f_tof_free_schroedinger, dtype=torch.float64,
                                                      device=self._device)
        self.z_f_tof_free_schroedinger = torch.tensor(z_f_tof_free_schroedinger, dtype=torch.float64,
                                                      device=self._device)

        self.index_center_x_f_tof_free_schroedinger = index_center_x_f_tof_free_schroedinger
        self.index_center_y_f_tof_free_schroedinger = index_center_y_f_tof_free_schroedinger
        self.index_center_z_f_tof_free_schroedinger = index_center_z_f_tof_free_schroedinger
        # ---------------------------------------------------------------------------------------------

    def compute_time_of_flight(self, **kwargs):

        print('----------------------------------------------------------------------------------------')
        print("time of flight:")

        self.psi_tof_free_gpe = qsolve_core_gpe_3d.init_psi_tof_free_gpe(
            self._psi,
            self.Jx_tof_free_gpe,
            self.Jy_tof_free_gpe,
            self.Jz_tof_free_gpe)

        print("propagate psi_tof_free_gpe ...")

        self.psi_tof_free_gpe = qsolve_core_gpe_3d.propagate_free_gpe(
            self.psi_tof_free_gpe,
            self.dx_tof_free_gpe,
            self.dy_tof_free_gpe,
            self.dz_tof_free_gpe,
            self.dt_tof_free_gpe,
            self.n_time_steps_tof_free_gpe,
            self._hbar,
            self._m_atom,
            self._g)

        self.psi_0_tof_free_schroedinger = self.psi_tof_free_gpe

        print("compute psi_tof_free_schroedinger ...")

        self.psi_f_tof_free_schroedinger = qsolve_core_gpe_3d.solve_tof_free_schroedinger(
            self.psi_0_tof_free_schroedinger,
            self.x_0_tof_free_schroedinger,
            self.y_0_tof_free_schroedinger,
            self.z_0_tof_free_schroedinger,
            self.x_f_tof_free_schroedinger,
            self.y_f_tof_free_schroedinger,
            self.z_f_tof_free_schroedinger,
            self.T_tof_free_schroedinger,
            self._hbar,
            self._m_atom)

        print('----------------------------------------------------------------------------------------')

        print()

    def propagate_sgpe_z_eff(self, **kwargs):

        self._psi = qsolve_core_gpe_3d.propagate_sgpe_z_eff(
            self._psi,
            self._V,
            self._dx,
            self._dy,
            self._dz,
            self.dt_sgpe,
            kwargs["n_inc"],
            self.T_des_sgpe,
            self.mue_des_sgpe,
            self.gamma_sgpe,
            self._hbar,
            self._k_B,
            self._m_atom,
            self._g)

        self._psi = self.filter_z_sgpe * self._psi

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

        if identifier == "times":

            if units == "si_units":

                return self._units.unit_time * self.times

            else:

                return self.times

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

        else:

            message = 'get(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

            raise Exception(message)

    def compute_n_atoms(self):
        return qsolve_core_gpe_3d.compute_n_atoms(self._psi, self._dx, self._dy, self._dz)

    def compute_chemical_potential(self):

        _mue = qsolve_core_gpe_3d.compute_chemical_potential(
            self._psi, self._V, self._dx, self._dy, self._dz, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _mue

    def compute_total_energy(self):

        _E = qsolve_core_gpe_3d.compute_total_energy(
            self._psi, self._V, self._dx, self._dy, self._dz, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _E

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core_gpe_3d.compute_kinetic_energy(
            self._psi, self._dx, self._dy, self._dz, self._hbar, self._m_atom)

        return self._units.unit_energy * _E_kinetic


    def compute_E_potential(self, identifier, **kwargs):

        if "units" in kwargs:

            units = kwargs["units"]

        else:

            units = "si_units"

        if identifier == "psi":

            E_potential = qsolve_core_gpe_3d.compute_potential_energy(self._psi, self._V, self._dx, self._dy, self._dz)

        elif identifier == "psi_0":

            E_potential = qsolve_core_gpe_3d.compute_potential_energy(self.psi_0, self._V, self._dx, self._dy, self._dz)

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

            E_interaction = qsolve_core_gpe_3d.compute_interaction_energy(
                self._psi,
                self._dx,
                self._dy,
                self._dz,
                self._g)

        elif identifier == "psi_0":

            E_interaction = qsolve_core_gpe_3d.compute_interaction_energy(
                self.psi_0,
                self._dx,
                self._dy,
                self._dz,
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

            return self._units.unit_energy * E_interaction

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

    @property
    def V(self):
        return self._units.unit_energy * self._V.cpu().numpy()

    @property
    def psi(self):
        return self._units.unit_wave_function * self._psi.cpu().numpy()

    @psi.setter
    def psi(self, value):
        self._psi = torch.tensor(value / self._units.unit_wave_function, device=self._device)

    @property
    def x(self):
        return self._units.unit_length * self._x.cpu().numpy()

    @property
    def y(self):
        return self._units.unit_length * self._y.cpu().numpy()

    @property
    def z(self):
        return self._units.unit_length * self._z.cpu().numpy()

    @property
    def dx(self):
        return self._units.unit_length * self._dx

    @property
    def dy(self):
        return self._units.unit_length * self._dy

    @property
    def dz(self):
        return self._units.unit_length * self._dz

    @property
    def Jx(self):
        return self._Jx

    @property
    def Jy(self):
        return self._Jy

    @property
    def Jz(self):
        return self._Jz

    @property
    def index_center_x(self):
        return self._index_center_x

    @property
    def index_center_y(self):
        return self._index_center_y

    @property
    def index_center_z(self):
        return self._index_center_z
