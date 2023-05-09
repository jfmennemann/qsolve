import numpy as np


def compute_external_potential(x, t, u, p, q):

    nu_start = p["nu_start"]
    nu_final = p["nu_final"]

    m_atom = q["m_atom"]

    omega_start = 2 * np.pi * nu_start
    omega_final = 2 * np.pi * nu_final

    u = u[0]

    omega = omega_start + u * (omega_final - omega_start)

    V = 0.5 * m_atom * omega ** 2 * x ** 2

    return V
