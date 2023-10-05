import numpy as np


class FigExcitationsV2D(object):

    def __init__(self, ax, V, excitations_v, nr, x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap):

        x = x / 1e-6
        y = y / 1e-6

        if np.max(np.abs(V)) > 0.0:

            V = V / np.max(np.abs(V))

        v = excitations_v[nr, :, :]

        Jx = x.size
        Jy = y.size

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
        # -----------------------------------------------------------------------------------------

        Y, X = np.meshgrid(x, y, indexing='ij')

        # Z = V / np.max(np.abs(V))

        ax.contour(X, Y, V, levels_V, colors='gray', linewidths=1.25)


        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        title = r'$v_{' + '{0:d}'.format(nr + 1) + '}(x, y)$'

        ax.set_title(title, fontsize=10)
