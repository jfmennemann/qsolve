import numpy as np

from qsolve.core import qsolve_core


class Grid2D(object):

    def __init__(self, *, x_min, x_max, y_min, y_max, Jx, Jy):

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.Jx = Jx
        self.Jy = Jy

        prime_factors_Jx = qsolve_core.get_prime_factors(self.Jx)
        prime_factors_Jy = qsolve_core.get_prime_factors(self.Jy)

        assert (np.max(prime_factors_Jx) < 11)
        assert (np.max(prime_factors_Jy) < 11)

        assert (self.Jx % 2 == 0)
        assert (self.Jy % 2 == 0)

        self.x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)
        self.y = np.linspace(self.y_min, self.y_max, self.Jy, endpoint=False)

        self.index_center_x = np.argmin(np.abs(self.x))
        self.index_center_y = np.argmin(np.abs(self.y))

        assert (np.abs(self.x[self.index_center_x]) < 1e-14)
        assert (np.abs(self.y[self.index_center_y]) < 1e-14)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.Lx = self.Jx * self.dx
        self.Ly = self.Jy * self.dy

        self.x_2d = np.reshape(self.x, newshape=(self.Jx, 1))
        self.y_2d = np.reshape(self.y, newshape=(1, self.Jy))
