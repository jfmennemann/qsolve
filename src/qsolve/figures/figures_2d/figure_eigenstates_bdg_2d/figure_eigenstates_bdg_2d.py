import matplotlib.pyplot as plt

import numpy as np

from .fig_excitation_u_2d import FigExcitationsU2D
from .fig_excitation_v_2d import FigExcitationsV2D

# from qsolve.visualization.colormaps import cmap_seaweed as cmap
from qsolve.visualization.colormaps import cmap_iceburn as cmap

# cmap = plt.get_cmap('RdBu')


class FigureEigenstatesBDG2D(object):

    def __init__(self, *,
                 excitations_u,
                 excitations_v,
                 V,
                 x,
                 y,
                 x_ticks,
                 y_ticks,
                 figsize=(12, 10),
                 left=0.065, right=0.975,
                 bottom=0.08, top=0.95,
                 wspace=0.35, hspace=0.7,
                 width_ratios=None,
                 height_ratios=None,
                 name="figure_eigenstates_bdg_2d"
                 ):

        if height_ratios is None:
            height_ratios = [1, 1, 1, 1, 1]

        if width_ratios is None:
            width_ratios = [1, 1, 1, 1]

        label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

        plt.rcParams.update({'font.size': 10})

        # -----------------------------------------------------------------------------------------
        self.fig_name = name

        self.fig = plt.figure(self.fig_name, figsize=figsize, facecolor="white")
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.gridspec = self.fig.add_gridspec(nrows=5, ncols=4,
                                              left=left, right=right,
                                              bottom=bottom, top=top,
                                              wspace=wspace,
                                              hspace=hspace,
                                              width_ratios=width_ratios,
                                              height_ratios=height_ratios)

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

        ax_02 = self.fig.add_subplot(self.gridspec[0, 2])
        ax_12 = self.fig.add_subplot(self.gridspec[1, 2])
        ax_22 = self.fig.add_subplot(self.gridspec[2, 2])
        ax_32 = self.fig.add_subplot(self.gridspec[3, 2])
        ax_42 = self.fig.add_subplot(self.gridspec[4, 2])

        ax_03 = self.fig.add_subplot(self.gridspec[0, 3])
        ax_13 = self.fig.add_subplot(self.gridspec[1, 3])
        ax_23 = self.fig.add_subplot(self.gridspec[2, 3])
        ax_33 = self.fig.add_subplot(self.gridspec[3, 3])
        ax_43 = self.fig.add_subplot(self.gridspec[4, 3])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        tmp = np.max(np.abs(excitations_u), axis=(1, 2), keepdims=True)

        excitations_u = excitations_u / tmp
        excitations_v = excitations_v / tmp

        levels_V = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        n = excitations_u.shape[0]

        nrs = [0, 1, 2, 3, 4, n-5, n-4, n-3, n-2, n-1]

        FigExcitationsU2D(ax_00, V, excitations_u, nrs[0], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_10, V, excitations_u, nrs[1], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_20, V, excitations_u, nrs[2], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_30, V, excitations_u, nrs[3], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_40, V, excitations_u, nrs[4], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)

        FigExcitationsV2D(ax_01, V, excitations_v, nrs[0], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_11, V, excitations_v, nrs[1], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_21, V, excitations_v, nrs[2], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_31, V, excitations_v, nrs[3], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_41, V, excitations_v, nrs[4], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)

        FigExcitationsU2D(ax_02, V, excitations_u, nrs[5], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_12, V, excitations_u, nrs[6], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_22, V, excitations_u, nrs[7], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_32, V, excitations_u, nrs[8], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsU2D(ax_42, V, excitations_u, nrs[9], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)

        FigExcitationsV2D(ax_03, V, excitations_v, nrs[5], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_13, V, excitations_v, nrs[6], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_23, V, excitations_v, nrs[7], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_33, V, excitations_v, nrs[8], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigExcitationsV2D(ax_43, V, excitations_v, nrs[9], x, y, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
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
