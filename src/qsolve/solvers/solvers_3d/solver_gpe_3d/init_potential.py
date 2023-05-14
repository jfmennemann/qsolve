def init_potential(self, Potential, params_user):

    params_solver = {
        "x_3d": self.x_3d,
        "y_3d": self.y_3d,
        "z_3d": self.z_3d,
        "Lx": self.Lx,
        "Ly": self.Ly,
        "Lz": self.Lz,
        "hbar": self.hbar,
        "mu_B": self.mu_B,
        "m_atom": self.m_atom,
        "unit_length": self._units.unit_length,
        "unit_time": self._units.unit_time,
        "unit_mass": self._units.unit_mass,
        "unit_energy": self._units.unit_energy,
        "unit_frequency": self._units.unit_frequency,
        "device": self.device
    }

    self.potential = Potential(params_solver, params_user)
