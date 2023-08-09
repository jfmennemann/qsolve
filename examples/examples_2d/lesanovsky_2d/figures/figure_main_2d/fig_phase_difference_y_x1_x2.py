import numpy as np

from qsolve.figures.style import colors


class FigPhaseDifferenceYX1X2(object):

    def __init__(self, ax, settings):

        y = settings.y

        indices_y_restr = settings.indices_y_restr

        y_restr = y[indices_y_restr]

        self.line_phase_difference_y, = ax.plot(y[indices_y_restr], np.zeros_like(y_restr), linewidth=1.00, linestyle='-', color=colors.wet_asphalt, label=r'single run')

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(r'$\Delta \, \phi\; / \; \pi$')

        ax.set_xticks(settings.y_ticks)

        ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8], minor=False)

        ax.set_yticks([-0.6, -0.2, 0.2, 0.6], minor=True)

        ax.set_xlim(settings.y_min, settings.y_max)

        phase_difference_min = -0.8
        phase_difference_max = +0.8

        ax.set_ylim(phase_difference_min - 0.1 * (phase_difference_max - phase_difference_min), phase_difference_max + 0.1 * (phase_difference_max - phase_difference_min))

        ax.grid(visible=True, which='major', color=colors.color_gridlines_major, linestyle='-', linewidth=0.5)
        ax.grid(visible=True, which='minor', color=colors.color_gridlines_minor, linestyle='-', linewidth=0.5, alpha=0.2)

        self.indices_y_restr = indices_y_restr

    def update(self, phase_difference_y_x1_x2):

        indices_y_restr = self.indices_y_restr

        self.line_phase_difference_y.set_ydata(phase_difference_y_x1_x2[indices_y_restr]/np.pi)
