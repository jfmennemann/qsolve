import torch

import numpy as np

from qsolve.utils.primes import get_prime_factors


def init_grid(self, kwargs):

    self.x_min = kwargs['x_min'] / self.units.unit_length
    self.x_max = kwargs['x_max'] / self.units.unit_length

    self.Jx = kwargs['Jx']

    prime_factors_Jx = get_prime_factors(self.Jx)

    assert (np.max(prime_factors_Jx) < 11)

    assert (self.Jx % 2 == 0)

    x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)

    self.index_center_x = np.argmin(np.abs(x))

    assert (np.abs(x[self.index_center_x]) < 1e-14)

    self.dx = x[1] - x[0]

    self.Lx = self.Jx * self.dx

    self.x = torch.tensor(x, dtype=torch.float64, device=self.device)
