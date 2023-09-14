import matplotlib.pyplot as plt

from scipy import constants

import numpy as np

from .fig_potential_2d import FigPotential2D
from .fig_real_part_2d import FigRealPart2D

from qsolve.figures.style import colors


class FigureEigenstatesLSE2D(object):

    def __init__(self, *, eigenstates_lse, V, x, y, x_ticks, y_ticks):

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

        # -----------------------------------------------------------------------------------------
        settings = type('', (), {})()

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

        self.gridspec = self.fig.add_gridspec(nrows=5, ncols=2,
                                              left=0.1, right=0.95,
                                              bottom=0.08, top=0.95,
                                              wspace=0.35,
                                              hspace=0.7,
                                              width_ratios=[1, 1],
                                              height_ratios=[1, 1, 1, 1, 1])

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])

        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])
        ax_20 = self.fig.add_subplot(self.gridspec[2, 0])
        ax_30 = self.fig.add_subplot(self.gridspec[3, 0])
        ax_40 = self.fig.add_subplot(self.gridspec[4, 0])

        ax_01 = self.fig.add_subplot(self.gridspec[0, 1])
        ax_11 = self.fig.add_subplot(self.gridspec[1, 1])
        ax_21 = self.fig.add_subplot(self.gridspec[2, 1])
        ax_31 = self.fig.add_subplot(self.gridspec[3, 1])
        ax_41 = self.fig.add_subplot(self.gridspec[4, 1])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        FigPotential2D(ax_00, V, settings)

        n = eigenstates_lse.shape[0]

        nrs = [0, 1, 2, 3, 4, n-4, n-3, n-2, n-1]

        FigRealPart2D(ax_10, eigenstates_lse[nrs[0], :, :], settings)
        FigRealPart2D(ax_20, eigenstates_lse[nrs[1], :, :], settings)
        FigRealPart2D(ax_30, eigenstates_lse[nrs[2], :, :], settings)
        FigRealPart2D(ax_40, eigenstates_lse[nrs[3], :, :], settings)

        FigRealPart2D(ax_01, eigenstates_lse[nrs[4], :, :], settings)
        FigRealPart2D(ax_11, eigenstates_lse[nrs[5], :, :], settings)
        FigRealPart2D(ax_21, eigenstates_lse[nrs[6], :, :], settings)
        FigRealPart2D(ax_31, eigenstates_lse[nrs[7], :, :], settings)
        FigRealPart2D(ax_41, eigenstates_lse[nrs[8], :, :], settings)
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
