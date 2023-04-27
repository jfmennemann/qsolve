class Potential(object):

    def __init__(self, params_solver, params_user):

        x_2d = params_solver["x_2d"]
        y_2d = params_solver["y_2d"]

        m_atom = params_solver["m_atom"]

        unit_frequency = params_solver["unit_frequency"]

        omega_x = params_user["omega_x"] / unit_frequency
        omega_y = params_user["omega_y"] / unit_frequency

        self.V = 0.5 * m_atom * (omega_x**2 * x_2d**2 + omega_y**2 * y_2d**2)

    def eval(self, u):

        return self.V
