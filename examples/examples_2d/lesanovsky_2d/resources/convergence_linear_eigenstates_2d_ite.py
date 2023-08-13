# eigenstates_lse, energies_lse, matrix_res_batch, vec_iter = solver.compute_eigenstates_lse_ite(
#     n_eigenstates=64,
#     n_iter_max=1000,
#     tau_0=0.05e-3,
#     # tau_0=0.1e-3,
#     # propagation_method='trotter',
#     # propagation_method='strang',
#     # propagation_method='ite_4th',
#     # propagation_method='ite_6th',
#     propagation_method='ite_8th',
#     # propagation_method='ite_10th',
#     # propagation_method='ite_12th',
#     return_residuals=True)


# =================================================================================================
# show convergence of linear eigenstate computation
# =================================================================================================
"""
n_eigenstates_lse = matrix_res_batch.shape[0]

n_lines = n_eigenstates_lse

c = np.arange(0, n_lines)

cmap_tmp = mpl.colormaps['Spectral']

norm = mpl.colors.Normalize(vmin=0, vmax=n_lines-1)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_tmp)
cmap.set_array([])


fig_conv_lse = plt.figure(num="figure_convergence_lse_1d", figsize=(1.5*6, 1.5*4))

fig_conv_lse.subplots_adjust(left=0.1, right=1.0, bottom=0.125, top=0.925)

ax = fig_conv_lse.add_subplot(111)

ax.set_facecolor((0.15, 0.15, 0.15))

ax.set_yscale('log')

ax.set_title('linear eigenstate computation')

plt.grid(visible=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-', linewidth=0.5)
# plt.grid(visible=False, which='minor', color='k', linestyle='-', linewidth=0.25)

for nr in range(n_eigenstates_lse):
    plt.plot(vec_iter, matrix_res_batch[nr, :], linewidth=1.5, linestyle='-', color=cmap.to_rgba(nr))

ax.set_xlim(0, vec_iter[-1])
ax.set_ylim(1e-14, 1e0)

plt.xlabel(r'number of iterations', labelpad=12)
plt.ylabel(r'residual error', labelpad=12)

cbar = fig_conv_lse.colorbar(cmap, ax=ax, label=r'# eigenstate')

ticks_true = np.linspace(0, n_eigenstates_lse+1, 4)

cbar.ax.tick_params(length=6, pad=4, which="major")

fig_conv_lse.canvas.start_event_loop(0.001)

plt.draw()
"""