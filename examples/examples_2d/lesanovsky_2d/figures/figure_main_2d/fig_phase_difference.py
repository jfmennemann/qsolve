import matplotlib.pyplot as plt

import numpy as np

# cmap_phase = plt.get_cmap('PRGn')
cmap_phase = plt.get_cmap('RdBu')


class FigPhaseDifference(object):

    def __init__(self, ax, settings):

        Jx = settings.Jx
        Jy = settings.Jy

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        phase_difference = np.zeros((Jx, Jy))

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image = ax.imshow(
            phase_difference,
            extent=[left, right, bottom, top],
            cmap=cmap_phase,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=1,
            origin='lower')

        ax.set_title(r'phase difference (weighted)', fontsize=settings.fontsize_titles)

    def update(self, phase_difference):

        self.image.set_data(phase_difference)
