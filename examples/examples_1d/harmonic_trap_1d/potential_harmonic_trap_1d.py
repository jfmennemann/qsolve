def eval_V(x, u, p, t, psi):

    omega_start = p["omega_start"]
    omega_final = p["omega_final"]

    m_atom = p["m_atom"]

    u = u[0]

    assert(0 <= u <= 1)

    omega = omega_start + u * (omega_final - omega_start)

    V = 0.5 * m_atom * omega ** 2 * x ** 2

    return V
