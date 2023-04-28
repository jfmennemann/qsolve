import torch

from numpy import pi


class Potential(object):

    def __init__(self, params_solver, params_user):

        x_2d = params_solver["x_2d"]
        y_2d = params_solver["y_2d"]

        Ly = params_solver["Ly"]

        m_atom = params_solver["m_atom"]

        unit_frequency = params_solver["unit_frequency"]
        unit_energy = params_solver["unit_energy"]

        omega_x = params_user["omega_x"] / unit_frequency

        V_harmonic_x = 0.5 * m_atom * omega_x ** 2 * x_2d ** 2

        self.V_harmonic_xy = V_harmonic_x

        self.V_lattice_z_max = params_user["V_lattice_z_max"] / unit_energy

        m = params_user["V_lattice_z_m"]

        self.V_lattice_z = 0.5 * (torch.cos(2.0 * pi * m * y_2d / Ly) + 1.0)

    def eval(self, u):

        amplitude_V_lattice_z = u * self.V_lattice_z_max

        V = self.V_harmonic_xy + amplitude_V_lattice_z * self.V_lattice_z

        return V
