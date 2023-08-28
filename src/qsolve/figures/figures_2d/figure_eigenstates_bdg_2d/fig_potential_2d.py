import matplotlib.pyplot as plt

import numpy as np


class FigPotential2D(object):

    def __init__(self, ax, V, settings):

        x = settings.x
        y = settings.y

        Y, X = np.meshgrid(x, y, indexing='ij')

        Z = V / np.max(np.abs(V))

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        extent = [left, right, bottom, top]

        # norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
        # cmap = plt.get_cmap('RdBu')
        # cmap = cm.turbo

        cmap = plt.get_cmap('binary')

        ax.imshow(
            Z,
            extent=extent,
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            origin='lower')

        # extent = [left, right, bottom, top]

        # levels = np.linspace(start=0.0, stop=1.0, num=20, endpoint=True)
        # levels = np.logspace(start=0.0, stop=1.0, num=8, endpoint=True, base=10.0) / 10.0
        levels = [0.05, 0.1, 0.2, 0.4, 0.8]



        # cset1 = ax.contourf(X, Y, Z, levels, norm=norm, cmap=cmap.resampled(len(levels) - 1))
        # cset1 = ax.contourf(X, Y, Z, levels, cmap=cmap.resampled(len(levels) - 1))

        # cset2 = ax.contour(X, Y, Z, cset1.levels, colors='black', linewidths=0.5)
        cset2 = ax.contour(X, Y, Z, levels, colors='black', linewidths=0.5)

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        # We don't really need dashed contour lines to indicate negative
        # regions, so let's turn them off.

        # for c in cset2.collections:
        #     c.set_linestyle('solid')

        ax.set_title(r'$V$ (scaled)', fontsize=settings.fontsize_titles)
