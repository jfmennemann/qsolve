def init_potential(self, Potential, params_user):

    params_solver = {
        "x_2d": self.x_2d,
        "y_2d": self.y_2d,
        "Lx": self.Lx,
        "Ly": self.Ly,
        "hbar": self.hbar,
        "mu_B": self.mu_B,
        "m_atom": self.m_atom,
        "unit_length": self.units.unit_length,
        "unit_time": self.units.unit_time,
        "unit_mass": self.units.unit_mass,
        "unit_energy": self.units.unit_energy,
        "unit_frequency": self.units.unit_frequency,
        "device": self.device
    }

    self.potential = Potential(params_solver, params_user)
