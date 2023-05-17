import math


def compute_external_potential(x_3d, y_3d, z_3d, t, u, p):

    m_atom = p["m_atom"]

    omega_x = 2.0 * math.pi * p["nu_x"]
    omega_y = 2.0 * math.pi * p["nu_y"]
    omega_z = 2.0 * math.pi * p["nu_z"]

    return 0.5 * m_atom * (omega_x**2 * x_3d**2 + omega_y**2 * y_3d**2 + omega_z**2 * z_3d**2)
