import numpy as np


def get(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "seed":

        return self.seed

    elif identifier == "hbar":

        if units == "si_units":

            return self.units.unit_hbar * self.hbar

        else:

            return self.hbar

    elif identifier == "mu_B":

        if units == "si_units":

            return self.units.unit_bohr_magneton * self.mu_B

        else:

            return self.mu_B

    elif identifier == "k_B":

        if units == "si_units":

            return self.units.unit_k_B * self.k_B

        else:

            return self.k_B

    elif identifier == "a_s":

        if units == "si_units":

            return self.units.unit_length * self.a_s

        else:

            return self.a_s

    elif identifier == "m_atom":

        if units == "si_units":

            return self.units.unit_mass * self.m_atom

        else:

            return self.m_atom

    elif identifier == "g":

        if units == "si_units":

            return self.units.unit_g * self.g

        else:

            return self.g

    elif identifier == "Jx":

        return self.Jx

    elif identifier == "Jy":

        return self.Jy

    elif identifier == "index_center_x":

        return self.index_center_x

    elif identifier == "index_center_y":

        return self.index_center_y

    elif identifier == "x":

        x = self.x.cpu().numpy()

        if units == "si_units":

            return self.units.unit_length * x

        else:

            return x

    elif identifier == "y":

        y = self.y.cpu().numpy()

        if units == "si_units":

            return self.units.unit_length * y

        else:

            return y

    elif identifier == "dx":

        if units == "si_units":

            return self.units.unit_length * self.dx

        else:

            return self.dx

    elif identifier == "dy":

        if units == "si_units":

            return self.units.unit_length * self.dy

        else:

            return self.dy

    elif identifier == "Lx":

        if units == "si_units":

            return self.units.unit_length * self.Lx

        else:

            return self.Lx

    elif identifier == "Ly":

        if units == "si_units":

            return self.units.unit_length * self.Ly

        else:

            return self.Ly

    elif identifier == "times":

        if units == "si_units":

            return self.units.unit_time * self.times

        else:

            return self.times

    elif identifier == "V":

        V = self.V.cpu().numpy()

        if units == "si_units":

            return self.units.unit_energy * V

        else:

            return V

    elif identifier == "psi_0":

        psi_0 = self.psi_0.cpu().numpy()

        if units == "si_units":

            return self.units.unit_wave_function * psi_0

        else:

            return psi_0

    elif identifier == "psi":

        psi = self.psi.cpu().numpy()

        if units == "si_units":

            return self.units.unit_wave_function * psi

        else:

            return psi

    elif identifier == "filter_z_sgpe":

        filter_z_sgpe = self.filter_z_sgpe.cpu().numpy()

        filter_z_sgpe = np.squeeze(filter_z_sgpe)

        return filter_z_sgpe

    elif identifier == "psi_tof_free_gpe":

        psi_tof_free_gpe = self.psi_tof_free_gpe.cpu().numpy()

        if units == "si_units":

            return self.units.unit_wave_function * psi_tof_free_gpe

        else:

            return psi_tof_free_gpe

    elif identifier == "vec_res_ground_state_computation":

        return self.vec_res_ground_state_computation.cpu().numpy()

    elif identifier == "vec_iter_ground_state_computation":

        return self.vec_iter_ground_state_computation.cpu().numpy()

    else:

        message = 'get(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

        raise Exception(message)
