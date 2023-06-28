import torch

import scipy

import numpy as np

import sys

import math

from qsolve.core import qsolve_core

# import qsolve_core

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

        self._z_min = kwargs['z_min'] / self._units.unit_length
        self._z_max = kwargs['z_max'] / self._units.unit_length

        self._Jx = kwargs['Jx']
        self._Jy = kwargs['Jy']
        self._Jz = kwargs['Jz']

        assert (np.max(qsolve_core.get_prime_factors(self._Jx)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self._Jy)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self._Jz)) < 11)

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

        self._p['Lx'] = self._Lx
        self._p['Ly'] = self._Ly
        self._p['Lz'] = self._Lz

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

        self._V = self._compute_external_potential(self._x_3d, self._y_3d, self._z_3d, t, u, self._p)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau=True, return_residuals=False):

        _tau = tau / self._units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core.ground_state_gpe_3d(
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

        self._T_des_sgpe = kwargs["T_temp_des"] / self._units.unit_temperature
        self._mue_des_sgpe = kwargs["mue_des"] / self._units.unit_energy
        self._gamma_sgpe = kwargs["gamma"]
        self._dt_sgpe = kwargs["dt"] / self._units.unit_time

        z1 = kwargs["filter_z1"] / self._units.unit_length
        z2 = kwargs["filter_z2"] / self._units.unit_length

        s = kwargs["filter_z_s"] / self._units.unit_length

        self.filter_z_sgpe = __compute_filter_z(self._z, z1, z2, s)

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

            self._V = self._compute_external_potential(self._x_3d, self._y_3d, self._z_3d, _t, u, self._p)

            self._psi = qsolve_core.propagate_gpe_3d(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                self._dz,
                _dt,
                _mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def init_time_of_flight(self, params):

        self._Jx_tof_free_gpe = params["Jx_tof_free_gpe"]
        self._Jy_tof_free_gpe = params["Jy_tof_free_gpe"]
        self._Jz_tof_free_gpe = params["Jz_tof_free_gpe"]

        self._T_tof_total = params["T_tof_total"] / self._units.unit_time
        self._T_tof_free_gpe = params["T_tof_free_gpe"] / self._units.unit_time

        self._T_tof_free_schroedinger = self._T_tof_total - self._T_tof_free_gpe

        self._dt_tof_free_gpe = params["dt_tof_free_gpe"] / self._units.unit_time

        self._dx_tof_free_gpe = self._dx
        self._dy_tof_free_gpe = self._dy
        self._dz_tof_free_gpe = self._dz

        # ---------------------------------------------------------------------------------------------
        self._n_time_steps_tof_free_gpe = int(np.round(self._T_tof_free_gpe / self._dt_tof_free_gpe))

        assert (self._n_time_steps_tof_free_gpe * self._dt_tof_free_gpe - self._T_tof_free_gpe) < 1e-14
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        assert (self._Jx_tof_free_gpe >= self._Jx)
        assert (self._Jy_tof_free_gpe >= self._Jy)
        assert (self._Jz_tof_free_gpe >= self._Jz)

        assert (self._Jx_tof_free_gpe % 2 == 0)
        assert (self._Jy_tof_free_gpe % 2 == 0)
        assert (self._Jz_tof_free_gpe % 2 == 0)

        assert (np.max(qsolve_core.get_prime_factors(self._Jx_tof_free_gpe)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self._Jy_tof_free_gpe)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self._Jz_tof_free_gpe)) < 11)
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        _x_tof_free_gpe = self._dx_tof_free_gpe * np.arange(-self._Jx_tof_free_gpe // 2, self._Jx_tof_free_gpe // 2)
        _y_tof_free_gpe = self._dy_tof_free_gpe * np.arange(-self._Jy_tof_free_gpe // 2, self._Jy_tof_free_gpe // 2)
        _z_tof_free_gpe = self._dz_tof_free_gpe * np.arange(-self._Jz_tof_free_gpe // 2, self._Jz_tof_free_gpe // 2)

        self._index_center_x_tof_free_gpe = np.argmin(np.abs(_x_tof_free_gpe))
        self._index_center_y_tof_free_gpe = np.argmin(np.abs(_y_tof_free_gpe))
        self._index_center_z_tof_free_gpe = np.argmin(np.abs(_z_tof_free_gpe))

        assert (np.abs(_x_tof_free_gpe[self._index_center_x_tof_free_gpe]) < 1e-14)
        assert (np.abs(_y_tof_free_gpe[self._index_center_y_tof_free_gpe]) < 1e-14)
        assert (np.abs(_z_tof_free_gpe[self._index_center_z_tof_free_gpe]) < 1e-14)

        self._x_tof_free_gpe = torch.tensor(_x_tof_free_gpe, dtype=torch.float64, device=self._device)
        self._y_tof_free_gpe = torch.tensor(_y_tof_free_gpe, dtype=torch.float64, device=self._device)
        self._z_tof_free_gpe = torch.tensor(_z_tof_free_gpe, dtype=torch.float64, device=self._device)
        # ---------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        self._x_0_tof_free_schroedinger = self._x_tof_free_gpe
        self._y_0_tof_free_schroedinger = self._y_tof_free_gpe
        self._z_0_tof_free_schroedinger = self._z_tof_free_gpe

        _Jx_tof_final = params["Jx_tof_final"]
        _Jy_tof_final = params["Jy_tof_final"]
        _Jz_tof_final = params["Jz_tof_final"]

        _x_min_tof_final = params["x_min_tof_final"] / self._units.unit_length
        _x_max_tof_final = params["x_max_tof_final"] / self._units.unit_length

        _y_min_tof_final = params["y_min_tof_final"] / self._units.unit_length
        _y_max_tof_final = params["y_max_tof_final"] / self._units.unit_length

        _z_min_tof_final = params["z_min_tof_final"] / self._units.unit_length
        _z_max_tof_final = params["z_max_tof_final"] / self._units.unit_length

        _x_f_tof_free_schroedinger = np.linspace(_x_min_tof_final, _x_max_tof_final, _Jx_tof_final, endpoint=True)
        _y_f_tof_free_schroedinger = np.linspace(_y_min_tof_final, _y_max_tof_final, _Jy_tof_final, endpoint=True)
        _z_f_tof_free_schroedinger = np.linspace(_z_min_tof_final, _z_max_tof_final, _Jz_tof_final, endpoint=True)

        _index_center_x_f_tof_free_schroedinger = np.argmin(np.abs(_x_f_tof_free_schroedinger))
        _index_center_y_f_tof_free_schroedinger = np.argmin(np.abs(_y_f_tof_free_schroedinger))
        _index_center_z_f_tof_free_schroedinger = np.argmin(np.abs(_z_f_tof_free_schroedinger))

        assert (np.abs(_x_f_tof_free_schroedinger[_index_center_x_f_tof_free_schroedinger]) < 5e-14)
        assert (np.abs(_y_f_tof_free_schroedinger[_index_center_y_f_tof_free_schroedinger]) < 5e-14)
        assert (np.abs(_z_f_tof_free_schroedinger[_index_center_z_f_tof_free_schroedinger]) < 5e-14)

        self._x_f_tof_free_schroedinger = torch.tensor(
            _x_f_tof_free_schroedinger, dtype=torch.float64, device=self._device)

        self._y_f_tof_free_schroedinger = torch.tensor(
            _y_f_tof_free_schroedinger, dtype=torch.float64, device=self._device)

        self._z_f_tof_free_schroedinger = torch.tensor(
            _z_f_tof_free_schroedinger, dtype=torch.float64, device=self._device)

        self._index_center_x_f_tof_free_schroedinger = _index_center_x_f_tof_free_schroedinger
        self._index_center_y_f_tof_free_schroedinger = _index_center_y_f_tof_free_schroedinger
        self._index_center_z_f_tof_free_schroedinger = _index_center_z_f_tof_free_schroedinger
        # ---------------------------------------------------------------------------------------------

    def compute_time_of_flight(self, **kwargs):

        print('----------------------------------------------------------------------------------------')
        print("time of flight:")

        self._psi_tof_free_gpe = qsolve_core.init_psi_tof_fgpe_3d(
            self._psi,
            self._Jx_tof_free_gpe,
            self._Jy_tof_free_gpe,
            self._Jz_tof_free_gpe)

        print("propagate psi_tof_free_gpe ...")

        self._psi_tof_free_gpe = qsolve_core.propagate_fgpe_3d(
            self._psi_tof_free_gpe,
            self._dx_tof_free_gpe,
            self._dy_tof_free_gpe,
            self._dz_tof_free_gpe,
            self._dt_tof_free_gpe,
            self._n_time_steps_tof_free_gpe,
            self._hbar,
            self._m_atom,
            self._g)

        self._psi_0_tof_free_schroedinger = self._psi_tof_free_gpe.clone()

        print("compute psi_tof_free_schroedinger ...")

        self._psi_f_tof_free_schroedinger = qsolve_core.solve_tof_fse_3d(
            self._psi_0_tof_free_schroedinger,
            self._x_0_tof_free_schroedinger,
            self._y_0_tof_free_schroedinger,
            self._z_0_tof_free_schroedinger,
            self._x_f_tof_free_schroedinger,
            self._y_f_tof_free_schroedinger,
            self._z_f_tof_free_schroedinger,
            self._T_tof_free_schroedinger,
            self._hbar,
            self._m_atom)

        print('----------------------------------------------------------------------------------------')

        print()

    def propagate_sgpe_z_eff(self, *, n_inc):

        self._psi = qsolve_core.propagate_sgpe_z_eff_3d(
            self._psi,
            self._V,
            self._dx,
            self._dy,
            self._dz,
            self._dt_sgpe,
            n_inc,
            self._T_des_sgpe,
            self._mue_des_sgpe,
            self._gamma_sgpe,
            self._hbar,
            self._k_B,
            self._m_atom,
            self._g)

        self._psi = self.filter_z_sgpe * self._psi

    def compute_n_atoms(self):
        return qsolve_core.n_atoms_3d(self._psi, self._dx, self._dy, self._dz)

    def compute_chemical_potential(self):

        _mue = qsolve_core.chemical_potential_gpe_3d(
            self._psi, self._V, self._dx, self._dy, self._dz, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _mue

    def compute_total_energy(self):

        _E = qsolve_core.total_energy_gpe_3d(
            self._psi, self._V, self._dx, self._dy, self._dz, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _E

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core.kinetic_energy_lse_3d(
            self._psi, self._dx, self._dy, self._dz, self._hbar, self._m_atom)

        return self._units.unit_energy * _E_kinetic

    def compute_potential_energy(self):

        _E_potential = qsolve_core.potential_energy_lse_3d(self._psi, self._V, self._dx, self._dy, self._dz)

        return self._units.unit_energy * _E_potential

    def compute_interaction_energy(self):

        _E_interaction = qsolve_core.interaction_energy_gpe_3d(self._psi, self._dx, self._dy, self._dz, self._g)

        return self._units.unit_energy * _E_interaction

    def trace_psi_tof_free_gpe_xy(self, index_z=None):

        if index_z is None:
            index_z = self._index_center_z_tof_free_gpe

        tmp = torch.squeeze(self._psi_tof_free_gpe[:, :, index_z])
        return self._units.unit_wave_function * tmp.cpu().numpy()

    def trace_psi_tof_free_gpe_xz(self, index_y=None):

        if index_y is None:
            index_y = self._index_center_y_tof_free_gpe

        tmp = torch.squeeze(self._psi_tof_free_gpe[:, index_y, :])

        return self._units.unit_wave_function * tmp.cpu().numpy()

    def trace_psi_f_tof_free_schroedinger_xy(self, index_z=None):

        if index_z is None:
            index_z = self._index_center_z_f_tof_free_schroedinger

        tmp = torch.squeeze(self._psi_f_tof_free_schroedinger[:, :, index_z])

        return self._units.unit_wave_function * tmp.cpu().numpy()

    def trace_psi_f_tof_free_schroedinger_xz(self, index_y=None):

        if index_y is None:
            index_y = self._index_center_y_f_tof_free_schroedinger

        tmp = torch.squeeze(self._psi_f_tof_free_schroedinger[:, index_y, :])

        return self._units.unit_wave_function * tmp.cpu().numpy()

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

    @property
    def x_tof_free_gpe(self):
        return self._units.unit_length * self._x_tof_free_gpe.cpu().numpy()

    @property
    def y_tof_free_gpe(self):
        return self._units.unit_length * self._y_tof_free_gpe.cpu().numpy()

    @property
    def z_tof_free_gpe(self):
        return self._units.unit_length * self._z_tof_free_gpe.cpu().numpy()

    @property
    def x_tof_final(self):
        return self._units.unit_length * self._x_f_tof_free_schroedinger.cpu().numpy()

    @property
    def y_tof_final(self):
        return self._units.unit_length * self._y_f_tof_free_schroedinger.cpu().numpy()

    @property
    def z_tof_final(self):
        return self._units.unit_length * self._z_f_tof_free_schroedinger.cpu().numpy()
