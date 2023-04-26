from qsolve.potentials.components_2d.harmonic_2d import eval_potential_harmonic_2d


class Potential(object):

    def __init__(self, params_solver, params_user):

        x_2d = params_solver["x_2d"]
        y_2d = params_solver["y_2d"]

        m_atom = params_solver["m_atom"]

        unit_frequency = params_solver["unit_frequency"]

        omega_x = params_user["omega_x"] / unit_frequency
        omega_y = params_user["omega_y"] / unit_frequency

        self.V = eval_potential_harmonic_2d(x_2d, y_2d, omega_x, omega_y, m_atom)

    def eval(self, u):

        return self.V
