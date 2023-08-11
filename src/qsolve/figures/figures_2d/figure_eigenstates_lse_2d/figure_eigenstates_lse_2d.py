import matplotlib.pyplot as plt

from scipy import constants

import numpy as np

from .fig_potential_2d import FigPotential2D
from .fig_density_2d import FigDensity2D

from qsolve.figures.style import colors


class FigureEigenstatesLSE2D(object):

    def __init__(self, eigenstates_lse, V, x, y, parameters):

        hbar = constants.hbar

        # m_atom = parameters['m_atom']

        # density_min = 0
        # density_max = parameters["density_max"]

        # V_min = parameters['V_min']
        # V_max = parameters['V_max']

        x = x / 1e-6
        y = y / 1e-6

        Jx = x.shape[0]
        Jy = y.shape[0]

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        x_min = x[0]
        y_min = y[0]

        x_max = x_min + Jx * dx
        y_max = y_min + Jy * dy

        Jx = x.size
        Jy = y.size

        # Lx = Jx * dx
        # Ly = Jy * dy

        x_ticks = parameters['x_ticks']
        y_ticks = parameters['y_ticks']

        t_ticks_major = parameters['t_ticks']

        # -----------------------------------------------------------------------------------------
        t_ticks_minor = 0.5 * (t_ticks_major[0:-1] + t_ticks_major[1:])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        settings = type('', (), {})()

        # settings.hbar = hbar
        # settings.m_atom = m_atom

        # settings.density_min = density_min
        # settings.density_max = density_max

        # settings.real_part_min = -1.2 * np.sqrt(settings.density_max)
        # settings.real_part_max = +1.2 * np.sqrt(settings.density_max)

        # settings.V_min = V_min
        # settings.V_max = V_max

        settings.x = x
        settings.y = y

        settings.Jx = Jx
        settings.Jy = Jy

        settings.x_ticks = x_ticks
        settings.y_ticks = y_ticks

        settings.x_min = x_min
        settings.x_max = x_max

        settings.y_min = y_min
        settings.y_max = y_max

        settings.t_ticks_major = t_ticks_major
        settings.t_ticks_minor = t_ticks_minor

        settings.label_V = r'$h \times \mathrm{kHz}$'

        settings.linecolor_V = colors.alizarin

        settings.linewidth_V = 1.1

        settings.label_density = r'$\mathrm{m}^{-2}$'

        settings.label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        settings.label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

        settings.label_t = r'$t \;\, \mathrm{in} \;\, \mathrm{ms}$'

        settings.cmap_density = colors.cmap_density

        settings.cmap_real_part = colors.cmap_real_part

        settings.color_gridlines_major = colors.color_gridlines_major
        settings.color_gridlines_minor = colors.color_gridlines_minor

        settings.fontsize_titles = 10
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.rcParams.update({'font.size': 10})
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_name = "figure_eigenstates_lse_2d"

        self.fig = plt.figure(self.fig_name, figsize=(8, 10), facecolor="white")
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # if Ly > Lx:
        #
        #     width_ratios = [1.25, 1, 2]
        #
        # elif Ly < Lx:
        #
        #     width_ratios = [1, 1.25, 2]
        #
        # else:
        #
        #     width_ratios = [1, 1, 2]

        self.gridspec = self.fig.add_gridspec(nrows=4, ncols=2,
                                              left=0.1, right=0.985,
                                              bottom=0.08, top=0.95,
                                              wspace=0.35,
                                              hspace=0.7,
                                              width_ratios=[1, 1],
                                              height_ratios=[1, 1, 1, 1])

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])
        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_potential_2d = FigPotential2D(ax_00, V, settings)

        self.fig_density_2d_10 = FigDensity2D(ax_10, np.abs(eigenstates_lse)**2, settings)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.ion()
        
        plt.draw()
        plt.pause(0.001)
        # -----------------------------------------------------------------------------------------

    def export(self, filepath):

        plt.figure(self.fig_name)

        plt.draw()

        self.fig.canvas.start_event_loop(0.001)

        plt.savefig(filepath,
                    dpi=None,
                    facecolor='w',
                    edgecolor='w',
                    format='png',
                    transparent=False,
                    bbox_inches=None,
                    pad_inches=0,
                    metadata=None)