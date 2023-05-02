def init_potential(self, Potential, params_user):

    params_solver = {
        "x": self.x,
        "Lx": self.Lx,
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
