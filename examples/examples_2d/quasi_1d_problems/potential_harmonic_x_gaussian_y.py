import torch


class Potential(object):

    def __init__(self, params_solver, params_user):

        x_2d = params_solver["x_2d"]
        y_2d = params_solver["y_2d"]

        m_atom = params_solver["m_atom"]

        unit_length = params_solver["unit_length"]
        unit_energy = params_solver["unit_energy"]
        unit_frequency = params_solver["unit_frequency"]

        omega_x = params_user["omega_x"] / unit_frequency

        sigma_gaussian_y = params_user["sigma_gaussian"] / unit_length

        self.V_harmonic_x = 0.5 * m_atom * omega_x**2 * x_2d**2

        self.V_gaussian_y = torch.exp(-y_2d**2 / (2 * sigma_gaussian_y**2))

        self.V_ref_gaussian = params_user["V_ref_gaussian"] / unit_energy

    def eval(self, u):

        amplitude_gaussian_y = u * self.V_ref_gaussian

        return self.V_harmonic_x + amplitude_gaussian_y * self.V_gaussian_y
