import numpy as np


class FigExcitationVXZ(object):

    def __init__(self, ax, V, excitations_v, nr, x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap):

        x = x / 1e-6
        y = y / 1e-6
        z = z / 1e-6

        if np.max(np.abs(V)) > 0.0:
            V = V / np.max(np.abs(V))

        Jx = V.shape[0]
        Jz = V.shape[2]

        index_y = np.argmin(np.abs(y))

        V = np.squeeze(V[:, index_y, :])
        v = np.squeeze(excitations_v[nr, :, index_y, :])

        dx = x[1] - x[0]
        dz = z[1] - z[0]

        x_min = x[0]
        z_min = z[0]

        x_max = x_min + Jx * dx
        z_max = z_min + Jz * dz

        if abs(x_max - x[-1]) > 1e-14:

            assert(abs(z_max - z[-1]) > 1e-14)

            x = np.append(x, x_max)
            z = np.append(z, z_max)

            # -------------------------------------------------------------------------------------
            V_new = np.zeros((Jx+1, Jz+1))

            V_new[0:Jx, 0:Jz] = V

            V_new[0, Jz] = V[0, 0]
            V_new[Jx, 0] = V[0, 0]
            V_new[Jx, Jz] = V[0, 0]

            V_new[:, Jz] = V_new[:, 0]
            V_new[Jx, :] = V_new[0, :]

            V = V_new
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            v_new = np.zeros((Jx + 1, Jz + 1))

            v_new[0:Jx, 0:Jz] = v

            v_new[0, Jz] = v[0, 0]
            v_new[Jx, 0] = v[0, 0]
            v_new[Jx, Jz] = v[0, 0]

            v_new[:, Jz] = v_new[:, 0]
            v_new[Jx, :] = v_new[0, :]

            v = v_new
            # -------------------------------------------------------------------------------------

        ax.set_xlabel(label_z)
        ax.set_ylabel(label_x)

        ax.set_xticks(z_ticks)
        ax.set_yticks(x_ticks)

        left = z_min
        right = z_max

        bottom = x_min
        top = x_max

        extent = [left, right, bottom, top]

        self.image = ax.imshow(
            v,
            extent=extent,
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        # -----------------------------------------------------------------------------------------
        # contour lines potential

        Z, X = np.meshgrid(x, z, indexing='ij')

        # levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # color_1 = (0.0, 0.0, 0.0)
        # color_2 = (0.0, 0.0, 0.0)
        # color_3 = (0.0, 0.0, 0.0)
        # color_4 = (0.0, 0.0, 0.0)
        # color_5 = (0.0, 0.0, 0.0)
        # color_6 = (0.0, 0.0, 0.0)
        # color_7 = (0.0, 0.0, 0.0)
        # color_8 = (0.0, 0.0, 0.0)
        # color_9 = (0.0, 0.0, 0.0)

        # colors = [color_1, color_2, color_3, color_4, color_5, color_6, color_7, color_8, color_9]

        # linewidths = np.linspace(start=0.15, stop=0.65, num=9)

        # linewidths = 0.25

        ax.contour(X, Z, V, levels_V, colors='gray', linewidths=1.25)
        # -----------------------------------------------------------------------------------------

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        # title = r'$v_{0:d}(x, 0, z)$'.format(nr)
        title = r'$v_{' + '{0:d}'.format(nr+1) + '}(x, 0, z)$'

        ax.set_title(title, fontsize=10)
