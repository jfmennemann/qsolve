class Potential(object):

    def __init__(self, params_solver, params_user):

        self.x = params_solver["x"]

        self.m_atom = params_solver["m_atom"]

        unit_frequency = params_solver["unit_frequency"]

        self.omega_start = params_user["omega_start"] / unit_frequency
        self.omega_final = params_user["omega_final"] / unit_frequency

    def eval(self, u):

        u = u[0]

        assert(0 <= u <= 1)

        omega = self.omega_start + u * (self.omega_final - self.omega_start)

        V = 0.5 * self.m_atom * omega ** 2 * self.x ** 2

        return u * V
