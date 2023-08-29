import numpy as np


class FigExcitationVXY(object):

    def __init__(self, ax, V, excitations_v, nr, x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap):

        x = x / 1e-6
        y = y / 1e-6
        z = z / 1e-6

        if np.max(np.abs(V)) > 0.0:
            V = V / np.max(np.abs(V))

        Jx = V.shape[0]
        Jy = V.shape[1]

        index_z = np.argmin(np.abs(z))

        V = np.squeeze(V[:, :, index_z])
        v = np.squeeze(excitations_v[nr, :, :, index_z])

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        x_min = x[0]
        y_min = y[0]

        x_max = x_min + Jx * dx
        y_max = y_min + Jy * dy

        if abs(x_max - x[-1]) > 1e-14:

            assert(abs(y_max - y[-1]) > 1e-14)

            x = np.append(x, x_max)
            y = np.append(y, y_max)

            # -------------------------------------------------------------------------------------
            V_new = np.zeros((Jx+1, Jy+1))

            V_new[0:Jx, 0:Jy] = V

            V_new[0, Jy] = V[0, 0]
            V_new[Jx, 0] = V[0, 0]
            V_new[Jx, Jy] = V[0, 0]

            V_new[:, Jy] = V_new[:, 0]
            V_new[Jx, :] = V_new[0, :]

            V = V_new
            # -------------------------------------------------------------------------------------

            # -------------------------------------------------------------------------------------
            v_new = np.zeros((Jx + 1, Jy + 1))

            v_new[0:Jx, 0:Jy] = v

            v_new[0, Jy] = v[0, 0]
            v_new[Jx, 0] = v[0, 0]
            v_new[Jx, Jy] = v[0, 0]

            v_new[:, Jy] = v_new[:, 0]
            v_new[Jx, :] = v_new[0, :]

            v = v_new
            # -------------------------------------------------------------------------------------

        ax.set_xlabel(label_y)
        ax.set_ylabel(label_x)

        ax.set_xticks(y_ticks)
        ax.set_yticks(x_ticks)

        left = y_min
        right = y_max

        bottom = x_min
        top = x_max

        extent = [left, right, bottom, top]

        self.image = ax.imshow(
            v / np.max(np.abs(v)),
            extent=extent,
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        # -----------------------------------------------------------------------------------------
        # contour lines potential

        Y, X = np.meshgrid(x, y, indexing='ij')

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

        ax.contour(X, Y, V, levels_V, colors='gray', linewidths=1.25)
        # -----------------------------------------------------------------------------------------

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        title = r'$v_{0:d}(x, y, 0)$'.format(nr)

        ax.set_title(title, fontsize=10)
