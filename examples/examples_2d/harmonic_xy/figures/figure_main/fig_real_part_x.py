from numpy import zeros_like

import numpy as np

from .. style import colors


class fig_real_part_x(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar
        self.m_atom = settings.m_atom

        # -----------------------------------------------------------------------------------------
        self.line_real_part_x, = ax.plot(settings.x, zeros_like(settings.x),
                                         linewidth=1, linestyle='-', color=colors.wet_asphalt,
                                         label=r'$\operatorname{Re}{\psi}$')
        self.line_imag_part_x, = ax.plot(settings.x, zeros_like(settings.x),
                                         linewidth=1, linestyle='--', color=colors.wet_asphalt,
                                         label=r'$\operatorname{Im}{\psi}$')

        ax.set_ylim(settings.real_part_min, settings.real_part_max)
        
        ax.set_xlabel(settings.label_x)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        ax.set_xticks(settings.x_ticks)
        
        ax.grid(visible=True, which='minor', color=settings.color_gridlines_minor, linestyle='-', linewidth=0.5, alpha=0.2)

        ax.set_ylabel(r'$\mathrm{m}^{-1}$')

        ax.set_title(r'$y=0$', fontsize=settings.fontsize_titles)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()
    
        self.line_V_x, = ax2.plot(settings.x, zeros_like(settings.x),
                                     linewidth=settings.linewidth_V, linestyle='-', color=settings.linecolor_V, label=r'$V$')

        ax2.set_xlim(settings.x_min, settings.x_max)
        ax2.set_ylim(settings.V_min, settings.V_max)
        
        ax2.set_ylabel(settings.label_V)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,
                   loc='upper right', bbox_to_anchor=(1.0, 1.25), fancybox=True, framealpha=1, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, real_part_x, imag_part_x, V_x):
        
        scaling_V = self.hbar * 2 * np.pi * 1000
        
        V_x = V_x / scaling_V

        self.line_real_part_x.set_ydata(real_part_x)
        self.line_imag_part_x.set_ydata(imag_part_x)
        
        self.line_V_x.set_ydata(V_x)
