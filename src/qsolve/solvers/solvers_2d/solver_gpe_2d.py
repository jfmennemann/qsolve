import torch

import scipy

import math

from qsolve.core import qsolve_core

# import qsolve_core


class SolverGPE2D(object):

    def __init__(self, *, units, grid, potential, device, m_atom, a_s, omega_z, seed=0, num_threads_cpu=1):

        self._potential = potential

        torch.manual_seed(seed)

        torch.set_num_threads(num_threads_cpu)

        self._device = torch.device(device)

        self._units = units

        # -----------------------------------------------------------------------------------------
        self._hbar = scipy.constants.hbar / self._units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self._m_atom = m_atom / self._units.unit_mass
        self._a_s = a_s / self._units.unit_length

        _omega_z = omega_z / self._units.unit_frequency

        _g_3d = 4.0 * math.pi * self._hbar ** 2 * self._a_s / self._m_atom

        _a_z = math.sqrt(self._hbar / (self._m_atom * _omega_z))

        self._g = _g_3d / (math.sqrt(2 * math.pi) * _a_z)

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        self._x = torch.tensor(grid.x / self._units.unit_length, device=self._device)
        self._y = torch.tensor(grid.y / self._units.unit_length, device=self._device)

        self._Lx = grid.Lx / self._units.unit_length
        self._Ly = grid.Ly / self._units.unit_length

        self._Jx = grid.Jx
        self._Jy = grid.Jy

        self._dx = grid.dx / self._units.unit_length
        self._dy = grid.dy / self._units.unit_length

        self._x_2d = torch.tensor(grid.x_2d / self._units.unit_length, device=self._device)
        self._y_2d = torch.tensor(grid.y_2d / self._units.unit_length, device=self._device)

        self._compute_external_potential = None
        self._V = None

        self._psi = None

    def set_external_potential(self, *, t, u):

        _t = t / self._units.unit_time

        _u = u

        self._V = self._potential.compute_external_potential(_t, _u)

    def compute_ground_state_solution(self, *, n_atoms, n_iter, tau, adaptive_tau=True, return_residuals=False):

        _tau = tau / self._units.unit_time

        if n_iter < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter should not be smaller than 2500'

            raise Exception(message)

        _psi_0, vec_res, vec_iter = qsolve_core.ground_state_gpe_2d(
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

    def eigenstates_lse_ite(self, *, n_eigenstates, tau_0, order):

        _tau_0 = tau_0 / self._units.unit_time

        _eigenstates, _eigenvalues = qsolve_core.compute_eigenstates_lse_2d_ite(
            self._V,
            self._dx,
            self._dy,
            self._hbar,
            self._m_atom,
            n_eigenstates,
            _tau_0,
            order
        )

        return self._units.unit_wave_function * _eigenstates.cpu().numpy(), self._units.unit_energy * _eigenvalues.cpu().numpy()

    def eigenstates_lse(self, *, n_eigenstates, tol=1e-12):

        _eigenstates, _eigenvalues = qsolve_core.compute_eigenstates_lse_2d(
            self._V,
            self._dx,
            self._dy,
            self._hbar,
            self._m_atom,
            n_eigenstates,
            tol
        )

        return \
                self._units.unit_wave_function * _eigenstates, \
                self._units.unit_energy * _eigenvalues

    def bdg(self, *, psi_0, n_atoms, n_excitations):

        _psi_0 = torch.tensor(psi_0 / self._units.unit_wave_function, device=self._device)

        _mue_0 = qsolve_core.chemical_potential_gpe_2d(
            _psi_0, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        _excitations_u, _excitations_v, _frequencies_omega, _psi_0, _mue_0, res_max = qsolve_core.bdg_2d(
            _psi_0,
            _mue_0,
            self._V,
            self._dx,
            self._dy,
            self._hbar,
            self._m_atom,
            self._g,
            n_atoms,
            n_excitations
        )

        return \
            self._units.unit_wave_function * _excitations_u, \
            self._units.unit_wave_function * _excitations_v, \
            self._units.unit_frequency * _frequencies_omega, \
            self._units.unit_wave_function * _psi_0, \
            self._units.unit_energy * _mue_0, \
            res_max

    def bdg_sse(self, *, psi_0, n_atoms, n_excitations, dim_subspace):

        _psi_0 = torch.tensor(psi_0 / self._units.unit_wave_function, device=self._device)

        _mue_0 = qsolve_core.chemical_potential_gpe_2d(
            _psi_0, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        _excitations_u, _excitations_v, _frequencies_omega, _psi_0, _mue_0, res_max = qsolve_core.bdg_2d_sse(
            _psi_0,
            _mue_0,
            self._V,
            self._dx,
            self._dy,
            self._hbar,
            self._m_atom,
            self._g,
            n_atoms,
            n_excitations,
            dim_subspace
        )

        return \
            self._units.unit_wave_function * _excitations_u, \
            self._units.unit_wave_function * _excitations_v, \
            self._units.unit_frequency * _frequencies_omega, \
            self._units.unit_wave_function * _psi_0, \
            self._units.unit_energy * _mue_0, \
            res_max

    def init_sgpe_z_eff(self, **kwargs):

        def __compute_filter_z(y, y1, y2, s):

            Jy = y.shape[0]

            filter_y_1st = 1.0 / (1.0 + torch.exp(-(y - y1) / s))
            filter_y_2nd = 1.0 / (1.0 + torch.exp((y - y2) / s))

            filter_y = filter_y_1st + filter_y_2nd - 1.0

            filter_y = torch.reshape(filter_y, shape=(1, Jy))

            return filter_y

        self._T_des_sgpe = kwargs["T_temp_des"] / self._units.unit_temperature
        self._mue_des_sgpe = kwargs["mue_des"] / self._units.unit_energy
        self._gamma_sgpe = kwargs["gamma"]
        self._dt_sgpe = kwargs["dt"] / self._units.unit_time

        y1 = kwargs["filter_y1"] / self._units.unit_length
        y2 = kwargs["filter_y2"] / self._units.unit_length

        s = kwargs["filter_y_s"] / self._units.unit_length

        self.filter_y_sgpe = __compute_filter_z(self._y, y1, y2, s)

    def propagate_sgpe_z_eff(self, *, n_inc):

        self._psi = qsolve_core.propagate_sgpe_z_eff_2d(
            self._psi,
            self._V,
            self._dx,
            self._dy,
            self._dt_sgpe,
            n_inc,
            self._T_des_sgpe,
            self._mue_des_sgpe,
            self._gamma_sgpe,
            self._hbar,
            self._k_B,
            self._m_atom,
            self._g)

        self._psi = self.filter_y_sgpe * self._psi

    def propagate_gpe(self, *, times, u_of_times, n_start, n_inc, mue_shift=0.0):

        _times = times / self._units.unit_time
        _dt = _times[1] - _times[0]

        _mue_shift = mue_shift / self._units.unit_energy

        n_local = 0

        while n_local < n_inc:

            n = n_start + n_local

            _t = _times[n]

            if u_of_times.ndim > 1:

                _u = 0.5 * (u_of_times[:, n] + u_of_times[:, n + 1])

            else:

                _u = 0.5 * (u_of_times[n] + u_of_times[n + 1])

            self._V = self._potential.compute_external_potential(_t, _u)

            self._psi = qsolve_core.propagate_gpe_2d(
                self._psi,
                self._V,
                self._dx,
                self._dy,
                _dt,
                _mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    def compute_n_atoms(self):
        return qsolve_core.n_atoms_2d(self._psi, self._dx, self._dy)

    def compute_chemical_potential(self):

        _mue = qsolve_core.chemical_potential_gpe_2d(
            self._psi, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _mue

    def compute_total_energy(self):

        _E = qsolve_core.total_energy_gpe_2d(
            self._psi, self._V, self._dx, self._dy, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _E

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core.kinetic_energy_lse_2d(self._psi, self._dx, self._dy, self._hbar, self._m_atom)

        return self._units.unit_energy * _E_kinetic

    def compute_potential_energy(self):

        _E_potential = qsolve_core.potential_energy_lse_2d(self._psi, self._V, self._dx, self._dy)

        return self._units.unit_energy * _E_potential

    def compute_interaction_energy(self):

        _E_interaction = qsolve_core.interaction_energy_gpe_2d(self._psi, self._dx, self._dy, self._g)

        return self._units.unit_energy * _E_interaction

    @property
    def V(self):
        return self._units.unit_energy * self._V.cpu().numpy()

    @property
    def psi(self):
        return self._units.unit_wave_function * self._psi.cpu().numpy()

    @psi.setter
    def psi(self, value):
        self._psi = torch.tensor(value / self._units.unit_wave_function, device=self._device)
