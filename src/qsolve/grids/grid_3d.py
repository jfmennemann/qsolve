import numpy as np

from qsolve.core import qsolve_core


class Grid3D(object):

    def __init__(self, *, x_min, x_max, y_min, y_max, z_min, z_max, Jx, Jy, Jz):

        assert (np.max(qsolve_core.get_prime_factors(Jx)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(Jy)) < 11)
        assert (np.max(qsolve_core.get_prime_factors(Jz)) < 11)

        assert (Jx % 2 == 0)
        assert (Jy % 2 == 0)
        assert (Jz % 2 == 0)

        x = np.linspace(x_min, x_max, Jx, endpoint=False)
        y = np.linspace(y_min, y_max, Jy, endpoint=False)
        z = np.linspace(z_min, z_max, Jz, endpoint=False)

        index_center_x = np.argmin(np.abs(x))
        index_center_y = np.argmin(np.abs(y))
        index_center_z = np.argmin(np.abs(z))

        assert (np.abs(x[index_center_x]) < 1e-14)
        assert (np.abs(y[index_center_y]) < 1e-14)
        assert (np.abs(z[index_center_z]) < 1e-14)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        Lx = Jx * dx
        Ly = Jy * dy
        Lz = Jz * dz

        x_3d = np.reshape(x, newshape=(Jx, 1, 1))
        y_3d = np.reshape(y, newshape=(1, Jy, 1))
        z_3d = np.reshape(z, newshape=(1, 1, Jz))

        self.x = x
        self.y = y
        self.z = z

        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

        self.index_center_x = index_center_x
        self.index_center_y = index_center_y
        self.index_center_z = index_center_z

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.z_min = z_min
        self.z_max = z_max

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.x_3d = x_3d
        self.y_3d = y_3d
        self.z_3d = z_3d
