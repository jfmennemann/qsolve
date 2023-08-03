from qsolve.potentials.components_3d.lesanovsky_xy_3d import eval_potential_lesanovsky_xy_3d
from qsolve.potentials.components_3d.box_z_3d import eval_potential_box_z_3d

import math


def compute_external_potential(x_3d, y_3d, z_3d, t, u, p):

    hbar = p["hbar"]
    mu_B = p["mu_B"]
    m_atom = p["m_atom"]

    g_F = p["g_F"]
    m_F = p["m_F"]
    m_F_prime = p["m_F_prime"]

    omega_perp = 2.0 * math.pi * p["nu_perp"]
    omega_para = 2.0 * math.pi * p["nu_para"]
    omega_delta_detuning = 2.0 * math.pi * p["nu_delta_detuning"]
    omega_trap_bottom = 2.0 * math.pi * p["nu_trap_bottom"]
    omega_rabi_ref = 2.0 * math.pi * p["nu_rabi_ref"]

    gamma_tilt_ref = p["gamma_tilt_ref"]

    V_box_z_max = p["V_box_z_max"]
    w_box_z = p["w_box_z"]
    s_box_z = p["s_box_z"]

    omega_rabi = u[0] * omega_rabi_ref

    V_lesanovsky = eval_potential_lesanovsky_xy_3d(
        x_3d,
        y_3d,
        z_3d,
        g_F,
        m_F,
        m_F_prime,
        omega_perp,
        omega_para,
        omega_delta_detuning,
        omega_trap_bottom,
        omega_rabi,
        hbar,
        mu_B,
        m_atom)

    V_tilt_x = -1.0 * u[1] * gamma_tilt_ref * x_3d

    z1 = w_box_z / 2.0
    z2 = -w_box_z / 2.0

    V_box_z = V_box_z_max * eval_potential_box_z_3d(z_3d, z1, z2, s_box_z)

    return V_lesanovsky + V_tilt_x + V_box_z
