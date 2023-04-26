from qsolve.core import qsolve_core_gpe_2d


def compute_n_atoms(self, identifier):

    if identifier == "psi":

        n_atoms = qsolve_core_gpe_2d.compute_n_atoms(self.psi, self.dx, self.dy)

    elif identifier == "psi_0":

        n_atoms = qsolve_core_gpe_2d.compute_n_atoms(self.psi_0, self.dx, self.dy)

    else:

        message = 'identifier \'{0:s}\' not supported for this operation'.format(identifier)

        raise Exception(message)

    return n_atoms
