import torch
import math


def compute_external_potential(x_3d, y_3d, z_3d, t, u, p):

    m_atom = p["m_atom"]

    omega_x = 2.0 * math.pi * p["nu_x"]
    omega_y = 2.0 * math.pi * p["nu_y"]

    V_lattice_z_max = p["V_lattice_z_max"]

    m = p["V_lattice_z_m"]

    Lz = p["Lz"]

    V_harmonic_xy = 0.5 * m_atom * (omega_x**2 * x_3d**2 + omega_y**2 * y_3d**2)

    V_lattice_z = 0.5 * (torch.cos(2.0 * math.pi * m * z_3d / Lz) + 1.0)

    return V_harmonic_xy + u * V_lattice_z_max * V_lattice_z
