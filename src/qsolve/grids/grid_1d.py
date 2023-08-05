import numpy as np

from qsolve.core import qsolve_core


class Grid1D(object):

    def __init__(self, *, x_min, x_max, Jx):

        self.x_min = x_min
        self.x_max = x_max

        self.Jx = Jx

        prime_factors_Jx = qsolve_core.get_prime_factors(self.Jx)

        assert (np.max(prime_factors_Jx) < 11)

        assert (self.Jx % 2 == 0)

        self.x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)

        self.index_center_x = np.argmin(np.abs(self.x))

        assert (np.abs(self.x[self.index_center_x]) < 1e-14)

        self.dx = self.x[1] - self.x[0]

        self.Lx = self.Jx * self.dx
