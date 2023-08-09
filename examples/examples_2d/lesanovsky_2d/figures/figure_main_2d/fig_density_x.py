import numpy as np

from qsolve.figures.style import colors


class fig_density_x(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar
        self.m_atom = settings.m_atom

        # -----------------------------------------------------------------------------------------
        self.line_density_x, = ax.plot(settings.x, np.zeros_like(settings.x),
                                       linewidth=1, linestyle='-', color=colors.wet_asphalt, label=r'$\rho(x, 0)$')

        density_min = settings.density_min
        density_max = settings.density_max

        ax.set_ylim(density_min - 0.1 * (density_max - density_min), density_max + 0.1 * (density_max - density_min))

        ax.set_yticks(np.linspace(density_min, density_max, num=3))

        ax.set_xlabel(settings.label_x)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        ax.set_xticks(settings.x_ticks)
        
        ax.grid(visible=True, which='minor', color=settings.color_gridlines_minor,
                linestyle='-', linewidth=0.5, alpha=0.2)
        
        ax.set_ylabel(settings.label_density)

        ax.set_title(r'$y=0$', fontsize=settings.fontsize_titles)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()
    
        self.line_V_x, = ax2.plot(settings.x, np.zeros_like(settings.x),
                                  linewidth=settings.linewidth_V, linestyle='-',
                                  color=settings.linecolor_V, label=r'$V(x, 0)$')

        ax2.set_xlim(settings.x_min, settings.x_max)

        V_min = settings.V_min
        V_max = settings.V_max

        ax2.set_ylim(V_min - 0.1 * (V_max - V_min), V_max + 0.1 * (V_max - V_min))

        ax2.set_yticks(np.linspace(V_min, V_max, num=3))
        
        ax2.set_ylabel(settings.label_V)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,
                   loc='upper right', bbox_to_anchor=(1.0, 1.25), fancybox=True, framealpha=1, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, density_x, V_x):
        
        scaling_V = self.hbar * 2 * np.pi * 1000
        
        V_x = V_x / scaling_V

        self.line_density_x.set_ydata(density_x)
        
        self.line_V_x.set_ydata(V_x)
