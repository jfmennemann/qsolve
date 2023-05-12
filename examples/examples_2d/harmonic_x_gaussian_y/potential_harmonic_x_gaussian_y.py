import torch
import math


def compute_external_potential(x_2d, y_2d, t, u, p):

    m_atom = p["m_atom"]

    nu_x = p["nu_x"]

    sigma_gaussian_y = p["sigma_gaussian_y"]

    omega_x = 2 * math.pi * nu_x

    V_harmonic_x = 0.5 * m_atom * omega_x ** 2 * x_2d ** 2

    V_gaussian_y = torch.exp(-y_2d ** 2 / (2 * sigma_gaussian_y ** 2))

    V_ref_gaussian_y = p["V_ref_gaussian_y"]

    amplitude_gaussian_y = u * V_ref_gaussian_y

    return V_harmonic_x + amplitude_gaussian_y * V_gaussian_y
