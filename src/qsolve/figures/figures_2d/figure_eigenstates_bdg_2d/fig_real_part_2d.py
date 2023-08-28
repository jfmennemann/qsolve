import matplotlib.pyplot as plt

import numpy as np


class FigRealPart2D(object):

    def __init__(self, ax, V, psi, settings, title):

        # -----------------------------------------------------------------------------------------
        # x = settings.x
        # y = settings.y
        #
        # Y, X = np.meshgrid(x, y, indexing='ij')
        #
        # Z = V / np.max(np.abs(V))
        # -----------------------------------------------------------------------------------------

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

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

        # levels = [0.05, 0.1, 0.2, 0.4, 0.8]
        #
        # ax.contour(X, Y, Z, levels, colors='black', linewidths=0.5)

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        ax.set_title(title, fontsize=settings.fontsize_titles)
