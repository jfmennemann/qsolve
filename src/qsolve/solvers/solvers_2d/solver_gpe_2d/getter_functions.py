import numpy as np


def get(self, identifier, kwargs):

    if "units" in kwargs:

        units = kwargs["units"]

    else:

        units = "si_units"

    if identifier == "psi":

        psi = self.psi.cpu().numpy()

        if units == "si_units":

            return self.units.unit_wave_function * psi

        else:

            return psi

    else:

        message = 'get(identifier, **kwargs): identifier \'{0:s}\' not supported'.format(identifier)

        raise Exception(message)
