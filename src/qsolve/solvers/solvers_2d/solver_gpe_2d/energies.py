from qsolve.core import qsolve_core_gpe_2d


def compute_E_interaction(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "psi":

        E_interaction = qsolve_core_gpe_2d.compute_interaction_energy(
            self.psi,
            self.dx,
            self.dy,
            self.g)

    elif identifier == "psi_0":

        E_interaction = qsolve_core_gpe_2d.compute_interaction_energy(
            self.psi_0,
            self.dx,
            self.dy,
            self.g)

    else:

        message = 'compute_E_interaction(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                  'not supported'.format(identifier)

        raise Exception(message)

    if units == "si_units":

        return self.units.unit_energy * E_interaction

    else:

        return E_interaction


def compute_E_kinetic(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "psi":

        E_kinetic = qsolve_core_gpe_2d.compute_kinetic_energy(
            self.psi,
            self.dx,
            self.dy,
            self.hbar,
            self.m_atom)

    elif identifier == "psi_0":

        E_kinetic = qsolve_core_gpe_2d.compute_kinetic_energy(
            self.psi_0,
            self.dx,
            self.dy,
            self.hbar,
            self.m_atom)

    else:

        message = 'compute_E_kinetic(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                  'not supported'.format(identifier)

        raise Exception(message)

    if units == "si_units":

        return self.units.unit_energy * E_kinetic

    else:

        return E_kinetic


def compute_E_potential(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "psi":

        E_potential = qsolve_core_gpe_2d.compute_potential_energy(self.psi, self.V, self.dx, self.dy)

    elif identifier == "psi_0":

        E_potential = qsolve_core_gpe_2d.compute_potential_energy(self.psi_0, self.V, self.dx, self.dy)

    else:

        message = 'compute_E_potential(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                  'not supported'.format(identifier)

        raise Exception(message)

    if units == "si_units":

        return self.units.unit_energy * E_potential

    else:

        return E_potential


def compute_E_total(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "psi":

        E = qsolve_core_gpe_2d.compute_total_energy(self.psi, self.V, self.dx, self.dy, self.hbar, self.m_atom, self.g)

    elif identifier == "psi_0":

        E = qsolve_core_gpe_2d.compute_total_energy(self.psi_0, self.V, self.dx, self.dy, self.hbar, self.m_atom, self.g)

    else:

        message = 'compute_E_total(self, identifier, **kwargs): \'identifier \'{0:s}\' ' \
                  'not supported'.format(identifier)

        raise Exception(message)

    if units == "si_units":

        return self.units.unit_energy * E

    else:

        return E
