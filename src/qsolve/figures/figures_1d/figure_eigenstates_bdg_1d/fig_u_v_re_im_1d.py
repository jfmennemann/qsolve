import numpy as np

from numpy import zeros_like

from numpy import pi

import math

from qsolve.figures.style import colors


class fig_u_v_re_im_1d(object):

    def __init__(self, ax, settings, legend=True):

        self.hbar = settings.hbar

        # -----------------------------------------------------------------------------------------
        self.line_u_re, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='-',
            color=colors.wet_asphalt, label=r'$\operatorname{Re}{u}$')

        self.line_u_im, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='--',
            color=colors.wet_asphalt, label=r'$\operatorname{Im}{u}$')

        self.line_v_re, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='-',
            color=colors.belize_hole, label=r'$\operatorname{Re}{v}$')

        self.line_v_im, = ax.plot(
            settings.x, zeros_like(settings.x), linewidth=1, linestyle='--',
            color=colors.belize_hole, label=r'$\operatorname{Im}{v}$')

        ax.set_xlim(settings.x_min, settings.x_max)

        u_v_re_im_min = settings.u_v_re_im_min
        u_v_re_im_max = settings.u_v_re_im_max

        ax.set_ylim(u_v_re_im_min - 0.1 * (u_v_re_im_max - u_v_re_im_min), u_v_re_im_max + 0.1 * (u_v_re_im_max - u_v_re_im_min))

        ax.set_yticks(np.linspace(u_v_re_im_min, u_v_re_im_max, num=5))

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

        ax2.set_yticks(np.linspace(V_min, V_max, num=5))
        
        ax2.set_ylabel(r'$h \times \mathrm{kHz}$', labelpad=10)
        # -----------------------------------------------------------------------------------------

        if legend:

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax2.legend(lines + lines2, labels + labels2,
                       loc='upper right', bbox_to_anchor=(1.0, 1.3), fancybox=True, framealpha=1, ncol=1)

    def update(self, u, v, V):

        u_re = np.real(u)
        u_im = np.imag(u)

        v_re = np.real(v)
        v_im = np.imag(v)
        
        scaling_V = self.hbar * 2 * pi * 1000
        
        V = V / scaling_V
        
        self.line_u_re.set_ydata(u_re / math.sqrt(1e6))
        self.line_u_im.set_ydata(u_im / math.sqrt(1e6))

        self.line_v_re.set_ydata(v_re / math.sqrt(1e6))
        self.line_v_im.set_ydata(v_im / math.sqrt(1e6))

        self.line_V.set_ydata(V)
