import matplotlib.pyplot as plt

import numpy as np


class FigRealPart2D(object):

    def __init__(self, ax, psi, settings):

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image = ax.imshow(
            psi / np.max(np.abs(psi)),
            extent=[left, right, bottom, top],
            # cmap=plt.get_cmap('PRGn'),
            cmap=plt.get_cmap('RdBu'),
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        ax.set_title(r'$\Re \psi$ (scaled)', fontsize=settings.fontsize_titles)
