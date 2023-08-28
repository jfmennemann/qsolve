import matplotlib.pyplot as plt

import numpy as np


class FigGroundStateGPE2D(object):

    def __init__(self, ax, V, psi_0, x, y, label_x, label_y, x_ticks, y_ticks, title):

        x = x / 1e-6
        y = y / 1e-6

        psi = psi_0

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
            psi_new = np.zeros((Jx + 1, Jy + 1))

            psi_new[0:Jx, 0:Jy] = psi

            psi_new[0, Jy] = psi[0, 0]
            psi_new[Jx, 0] = psi[0, 0]
            psi_new[Jx, Jy] = psi[0, 0]

            psi_new[:, Jy] = psi_new[:, 0]
            psi_new[Jx, :] = psi_new[0, :]

            psi = psi_new
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
            psi / np.max(np.abs(psi)),
            extent=extent,
            cmap=plt.get_cmap('RdBu'),
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        # -----------------------------------------------------------------------------------------
        # contour lines potential

        Y, X = np.meshgrid(x, y, indexing='ij')

        Z = V / np.max(np.abs(V))

        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        ax.contour(X, Y, Z, levels, colors='black', linewidths=0.25)
        # -----------------------------------------------------------------------------------------

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        ax.set_title(title, fontsize=10)
