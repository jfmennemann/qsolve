import torch

import scipy

import sys

import math

from qsolve.core import qsolve_core

from qsolve.units import Units


class SolverGPE1D(object):

    def __init__(self, *, grid, m_atom, a_s, omega_perp, seed=0, device='cpu', num_threads_cpu=1):

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

        self._units = Units.solver_units(m_atom, dim=1)

        # -----------------------------------------------------------------------------------------
        self._hbar = scipy.constants.hbar / self._units.unit_hbar
        self._mu_B = scipy.constants.physical_constants['Bohr magneton'][0] / self._units.unit_bohr_magneton
        self._k_B = scipy.constants.Boltzmann / self._units.unit_k_B

        self._m_atom = m_atom / self._units.unit_mass
        self._a_s = a_s / self._units.unit_length

        _omega_perp = omega_perp / self._units.unit_frequency

        _g_3d = 4.0 * scipy.constants.pi * self._hbar ** 2 * self._a_s / self._m_atom

        _a_perp = math.sqrt(self._hbar / (self._m_atom * _omega_perp))

        self._g = _g_3d / (2 * math.pi * _a_perp**2)

        assert (self._hbar == 1.0)
        assert (self._mu_B == 1.0)
        assert (self._k_B == 1.0)

        assert (self._m_atom == 1.0)
        # -----------------------------------------------------------------------------------------

        self._x = torch.tensor(grid.x / self._units.unit_length, device=self._device)

        self._x_min = grid.x_min / self._units.unit_length
        self._x_max = grid.x_max / self._units.unit_length

        self._Lx = grid.Lx / self._units.unit_length

        self._Jx = grid.Jx

        self._dx = grid.dx / self._units.unit_length

        self._index_center_x = grid.index_center_x

        self._compute_external_potential = None
        self._V = None

        self._psi = None

        self._p = {
            "hbar": self._hbar,
            "mu_B": self._mu_B,
            "k_B": self._k_B,
            "m_atom": self._m_atom
        }

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

        self._V = self._compute_external_potential(self._x, t, u, self._p)

    def compute_ground_state_solution(self, *, n_atoms, n_iter_max, tau_0, adaptive_tau=True, return_residuals=False):

        _tau_0 = tau_0 / self._units.unit_time

        if n_iter_max < 2500:

            message = 'compute_ground_state_solution(self, **kwargs): n_iter_max should not be smaller than 2500'

            raise Exception(message)

        _psi_0, _mue_0, vec_res, vec_iter = qsolve_core.ground_state_gpe_1d(
            self._V,
            self._dx,
            n_atoms,
            self._hbar,
            self._m_atom,
            self._g,
            n_iter_max,
            _tau_0,
            adaptive_tau,
            )

        if return_residuals:

            return \
                self._units.unit_wave_function * _psi_0.cpu().numpy(), \
                self._units.unit_energy * _mue_0, \
                vec_res, \
                vec_iter

        else:

            return \
                self._units.unit_wave_function * _psi_0.cpu().numpy(), \
                self._units.unit_energy * _mue_0

    def compute_eigenstates_lse(self,
                                *,
                                n_eigenstates,
                                n_iter_max,
                                tau_0,
                                propagation_method,
                                orthogonalization_method,
                                return_residuals=False):

        _tau_0 = tau_0 / self._units.unit_time

        # if n_iter_max < 100:
        #
        #     message = 'compute_ground_state_solution(self, **kwargs): n_iter_max should not be smaller than 2500'
        #
        #     raise Exception(message)

        _eigenstates_batch, _eigenvalues_batch, matrix_res_batch_of_vec_n_iter, vec_n_iter = qsolve_core.compute_eigenstates_lse_1d(
            self._V,
            self._dx,
            self._hbar,
            self._m_atom,
            n_eigenstates,
            n_iter_max,
            _tau_0,
            propagation_method,
            orthogonalization_method
        )

        if return_residuals:

            return \
                self._units.unit_wave_function * _eigenstates_batch.cpu().numpy(), \
                self._units.unit_energy * _eigenvalues_batch.cpu().numpy(), \
                matrix_res_batch_of_vec_n_iter, \
                vec_n_iter

        else:

            return \
                    self._units.unit_wave_function * _eigenstates_batch.cpu().numpy(), \
                    self._units.unit_energy * _eigenvalues_batch.cpu().numpy()

    def compute_eigenstates_lse_fd(self, *, n_eigenstates_max):

        _eigenstates_batch, _eigenvalues_batch = qsolve_core.compute_eigenstates_lse_1d_fd(
            self._V,
            self._dx,
            self._hbar,
            self._m_atom,
            n_eigenstates_max
        )

        return \
            self._units.unit_wave_function * _eigenstates_batch.cpu().numpy(), \
            self._units.unit_energy * _eigenvalues_batch.cpu().numpy()

        # return \
        #     self._units.unit_wave_function * _eigenstates_batch, \
        #     self._units.unit_energy * _eigenvalues_batch

    def bdg(self, *, n_atoms, n):

        # _psi_0 = torch.tensor(psi_0 / self._units.unit_wave_function, device=self._device)
        # _mue_0 = mue_0 / self._units.unit_energy

        _eigenvectors_u, _eigenvectors_v, _eigenvalues_omega, _psi_0, _mue_0 = qsolve_core.bdg_1d_sparse(
            self._V,
            self._dx,
            self._hbar,
            self._m_atom,
            self._g,
            n_atoms,
            n
        )

        # return \
        #     self._units.unit_wave_function * _eigenvectors_u.cpu().numpy(), \
        #     self._units.unit_wave_function * _eigenvectors_v.cpu().numpy(), \
        #     self._units.unit_frequency * _eigenvalues_omega.cpu().numpy(), \
        #     self._units.unit_wave_function * _psi_0.cpu().numpy(), \
        #     self._units.unit_energy * _mue_0

        return \
            self._units.unit_wave_function * _eigenvectors_u, \
            self._units.unit_wave_function * _eigenvectors_v, \
            self._units.unit_frequency * _eigenvalues_omega, \
            self._units.unit_wave_function * _psi_0, \
            self._units.unit_energy * _mue_0

    def init_sgpe(self, **kwargs):

        self._T_des_sgpe = kwargs["T_temp_des"] / self._units.unit_temperature
        self._mue_des_sgpe = kwargs["mue_des"] / self._units.unit_energy
        self._gamma_sgpe = kwargs["gamma"]
        self._dt_sgpe = kwargs["dt"] / self._units.unit_time

    def propagate_sgpe(self, *, n_inc):

        self._psi = qsolve_core.propagate_sgpe_1d(
            self._psi,
            self._V,
            self._dx,
            self._dt_sgpe,
            n_inc,
            self._T_des_sgpe,
            self._mue_des_sgpe,
            self._gamma_sgpe,
            self._hbar,
            self._k_B,
            self._m_atom,
            self._g)

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

            self._V = self._compute_external_potential(self._x, _t, _u, self._p)

            self._psi = qsolve_core.propagate_gpe_1d(
                self._psi,
                self._V,
                self._dx,
                _dt,
                _mue_shift,
                self._hbar,
                self._m_atom,
                self._g)

            n_local = n_local + 1

    @property
    def x(self):
        return self._units.unit_length * self._x.cpu().numpy()

    @property
    def dx(self):
        return self._units.unit_length * self._dx

    @property
    def psi(self):
        return self._units.unit_wave_function * self._psi.cpu().numpy()

    @psi.setter
    def psi(self, psi):
        self._psi = torch.tensor(psi / self._units.unit_wave_function, device=self._device)

    @property
    def V(self):
        return self._units.unit_energy * self._V.cpu().numpy()

    def compute_n_atoms(self):
        return qsolve_core.n_atoms_1d(self._psi, self._dx)

    def compute_kinetic_energy(self):

        _E_kinetic = qsolve_core.kinetic_energy_lse_1d(self._psi, self._dx, self._hbar, self._m_atom)

        return self._units.unit_energy * _E_kinetic

    def compute_potential_energy(self):

        _E_potential = qsolve_core.potential_energy_lse_1d(self._psi, self._V, self._dx)

        return self._units.unit_energy * _E_potential

    def compute_interaction_energy(self):

        _E_interaction = qsolve_core.interaction_energy_gpe_1d(self._psi, self._dx, self._g)

        return self._units.unit_energy * _E_interaction

    def compute_total_energy(self):

        _E = qsolve_core.total_energy_gpe_1d(
            self._psi, self._V, self._dx, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _E

    def compute_chemical_potential(self):

        _mue = qsolve_core.chemical_potential_gpe_1d(
            self._psi, self._V, self._dx, self._hbar, self._m_atom, self._g)

        return self._units.unit_energy * _mue
