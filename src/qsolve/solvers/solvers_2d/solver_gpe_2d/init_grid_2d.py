import torch

import numpy as np

from qsolve.primes import get_prime_factors


def init_grid(self, kwargs):

    self.x_min = kwargs['x_min'] / self.units.unit_length
    self.x_max = kwargs['x_max'] / self.units.unit_length

    self.y_min = kwargs['y_min'] / self.units.unit_length
    self.y_max = kwargs['y_max'] / self.units.unit_length

    self.Jx = kwargs['Jx']
    self.Jy = kwargs['Jy']

    prime_factors_Jx = get_prime_factors(self.Jx)
    prime_factors_Jy = get_prime_factors(self.Jy)

    assert (np.max(prime_factors_Jx) < 11)
    assert (np.max(prime_factors_Jy) < 11)

    assert (self.Jx % 2 == 0)
    assert (self.Jy % 2 == 0)

    x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)
    y = np.linspace(self.y_min, self.y_max, self.Jy, endpoint=False)

    self.index_center_x = np.argmin(np.abs(x))
    self.index_center_y = np.argmin(np.abs(y))

    assert (np.abs(x[self.index_center_x]) < 1e-14)
    assert (np.abs(y[self.index_center_y]) < 1e-14)

    self.dx = x[1] - x[0]
    self.dy = y[1] - y[0]

    self.Lx = self.Jx * self.dx
    self.Ly = self.Jy * self.dy

    self.x = torch.tensor(x, dtype=torch.float64, device=self.device)
    self.y = torch.tensor(y, dtype=torch.float64, device=self.device)

    self.x_2d = torch.reshape(self.x, (self.Jx, 1))
    self.y_2d = torch.reshape(self.y, (1, self.Jy))
