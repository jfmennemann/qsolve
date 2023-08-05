import numpy as np

from qsolve.core import qsolve_core


class Grid3D(object):

    def __init__(self, *, x_min, x_max, y_min, y_max, z_min, z_max, Jx, Jy, Jz):

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.z_min = z_min
        self.z_max = z_max

        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

        assert (np.max(qsolve_core.get_prime_factors(self.Jx)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self.Jy)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(self.Jz)) < 11)

        assert (self.Jx % 2 == 0)
        assert (self.Jy % 2 == 0)
        assert (self.Jz % 2 == 0)

        self.x = np.linspace(self.x_min, self.x_max, self.Jx, endpoint=False)
        self.y = np.linspace(self.y_min, self.y_max, self.Jy, endpoint=False)
        self.z = np.linspace(self.z_min, self.z_max, self.Jz, endpoint=False)

        self.index_center_x = np.argmin(np.abs(self.x))
        self.index_center_y = np.argmin(np.abs(self.y))
        self.index_center_z = np.argmin(np.abs(self.z))

        assert (np.abs(self.x[self.index_center_x]) < 1e-14)
        assert (np.abs(self.y[self.index_center_y]) < 1e-14)
        assert (np.abs(self.z[self.index_center_z]) < 1e-14)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.Lx = self.Jx * self.dx
        self.Ly = self.Jy * self.dy
        self.Lz = self.Jz * self.dz

        self.x_3d = np.reshape(self.x, newshape=(self.Jx, 1, 1))
        self.y_3d = np.reshape(self.y, newshape=(1, self.Jy, 1))
        self.z_3d = np.reshape(self.z, newshape=(1, 1, self.Jz))
