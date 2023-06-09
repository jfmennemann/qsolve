import numpy as np

from numpy import zeros_like

from numpy import pi

import math

from qsolve.figures.style import colors


class fig_psi_re_im_1d(object):

    def __init__(self, ax, settings, legend=True):

        self.hbar = settings.hbar

        # -----------------------------------------------------------------------------------------
        self.line_psi_re, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='-',
            color=colors.wet_asphalt, label=r'$\operatorname{Re}{\psi}$')

        self.line_psi_im, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='--',
            color=colors.wet_asphalt, label=r'$\operatorname{Im}{\psi}$')

        ax.set_xlim(settings.x_min, settings.x_max)

        psi_re_min = settings.psi_re_min
        psi_re_max = settings.psi_re_max

        ax.set_ylim(psi_re_min - 0.1 * (psi_re_max - psi_re_min), psi_re_max + 0.1 * (psi_re_max - psi_re_min))

        ax.set_yticks(np.linspace(psi_re_min, psi_re_max, 5))

        ax.set_xlabel(settings.label_x)
        
        ax.set_xticks(settings.x_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)

        ax.set_ylabel(r'$\mu \mathrm{m}^{-1 \, / \, 2}$', labelpad=0)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()

        self.line_V, = ax2.plot(
            settings.x, zeros_like(settings.x),
            linewidth=settings.linewidth_V, linestyle='-', color=settings.linecolor_V, label=r'$V$')

        ax2.set_xlim(settings.x_min, settings.x_max)

        V_min = settings.V_min
        V_max = settings.V_max

        ax2.set_ylim(V_min - 0.1 * (V_max - V_min), V_max + 0.1 * (V_max - V_min))

        ax2.set_yticks(np.linspace(V_min, V_max, 5))
        
        ax2.set_ylabel(r'$h \times \mathrm{kHz}$', labelpad=10)
        # -----------------------------------------------------------------------------------------

        if legend:

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax2.legend(lines + lines2, labels + labels2,
                       loc='upper right', bbox_to_anchor=(1.0, 1.3), fancybox=True, framealpha=1, ncol=1)

    def update(self, psi, V):

        psi_re = np.real(psi)
        psi_im = np.imag(psi)
        
        scaling_V = self.hbar * 2 * pi * 1000
        
        V = V / scaling_V
        
        self.line_psi_re.set_ydata(psi_re / math.sqrt(1e6))
        self.line_psi_im.set_ydata(psi_im / math.sqrt(1e6))

        self.line_V.set_ydata(V)
