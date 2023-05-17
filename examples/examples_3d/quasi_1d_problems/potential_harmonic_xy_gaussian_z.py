import torch
import math


def compute_external_potential(x_3d, y_3d, z_3d, t, u, p):

    m_atom = p["m_atom"]

    omega_x = 2.0 * math.pi * p["nu_x"]
    omega_y = 2.0 * math.pi * p["nu_y"]

    V_ref_gaussian_z = p['V_ref_gaussian_z']
    sigma_gaussian_z = p['sigma_gaussian_z']

    V_harmonic_xy = 0.5 * m_atom * (omega_x**2 * x_3d**2 + omega_y**2 * y_3d**2)

    V_gaussian_z = u * V_ref_gaussian_z * torch.exp(-z_3d ** 2 / (2 * sigma_gaussian_z ** 2))

    return V_harmonic_xy + V_gaussian_z
