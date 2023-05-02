from qsolve.core import qsolve_core_gpe_1d


def compute_n_atoms(self, identifier):

    if identifier == "psi":

        n_atoms = qsolve_core_gpe_1d.compute_n_atoms(self.psi, self.dx)

    elif identifier == "psi_0":

        n_atoms = qsolve_core_gpe_1d.compute_n_atoms(self.psi_0, self.dx)

    else:

        message = 'identifier \'{0:s}\' not supported for this operation'.format(identifier)

        raise Exception(message)

    return n_atoms
