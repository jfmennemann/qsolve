import matplotlib.pyplot as plt

import numpy as np

from .fig_eigenstate_psi_xz import FigEigenstatePsiXZ
from .fig_eigenstate_psi_xy import FigEigenstatePsiXY


# from qsolve.visualization.colormaps import cmap_seaweed as cmap
from qsolve.visualization.colormaps import cmap_iceburn as cmap


class FigureEigenstatesLSE3D(object):

    def __init__(self, *,
                 eigenstates,
                 V,
                 x,
                 y,
                 z,
                 x_ticks,
                 y_ticks,
                 z_ticks,
                 figsize=(12, 10),
                 left=0.1, right=0.95,
                 bottom=0.075, top=0.9,
                 wspace=0.35, hspace=0.7,
                 width_ratios=None,
                 height_ratios=None):

        if height_ratios is None:
            height_ratios = [1, 1, 1, 1, 1]

        if width_ratios is None:
            width_ratios = [1, 1, 1, 1]

        label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        label_z = r'$z \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

        plt.rcParams.update({'font.size': 10})

        # -----------------------------------------------------------------------------------------
        self.fig_name = "figure_eigenstates_bdg_3d"

        self.fig = plt.figure(self.fig_name, figsize=figsize, facecolor="white")

        self.fig.suptitle('eigenstates (linear Schrödinger equation)', fontsize=14)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.gridspec = self.fig.add_gridspec(nrows=5, ncols=4,
                                              left=left, right=right,
                                              bottom=bottom, top=top,
                                              wspace=wspace, hspace=hspace,
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
        # n = excitations_u.shape[0]

        # cmap = plt.get_cmap('RdBu')

        n = eigenstates.shape[0]

        tmp = np.max(np.abs(eigenstates), axis=(1, 2, 3), keepdims=True)

        eigenstates = eigenstates / tmp

        levels_V = np.linspace(start=0.1, stop=0.9, num=9, endpoint=True)

        nrs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # nrs = [0, 1, 2, 3, 4, n-5, n-4, n-3, n-2, n-1]

        FigEigenstatePsiXZ(ax_00, V, eigenstates, nrs[0], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_10, V, eigenstates, nrs[1], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_20, V, eigenstates, nrs[2], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_30, V, eigenstates, nrs[3], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_40, V, eigenstates, nrs[4], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)

        FigEigenstatePsiXY(ax_01, V, eigenstates, nrs[0], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_11, V, eigenstates, nrs[1], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_21, V, eigenstates, nrs[2], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_31, V, eigenstates, nrs[3], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_41, V, eigenstates, nrs[4], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)

        FigEigenstatePsiXZ(ax_02, V, eigenstates, nrs[5], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_12, V, eigenstates, nrs[6], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_22, V, eigenstates, nrs[7], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_32, V, eigenstates, nrs[8], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)
        FigEigenstatePsiXZ(ax_42, V, eigenstates, nrs[9], x, y, z, label_x, label_z, x_ticks, z_ticks, levels_V, cmap)

        FigEigenstatePsiXY(ax_03, V, eigenstates, nrs[5], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_13, V, eigenstates, nrs[6], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_23, V, eigenstates, nrs[7], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_33, V, eigenstates, nrs[8], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
        FigEigenstatePsiXY(ax_43, V, eigenstates, nrs[9], x, y, z, label_x, label_y, x_ticks, y_ticks, levels_V, cmap)
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
