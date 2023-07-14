"""
n_eigenstates_lse = matrix_res_batch.shape[1]

# cmap = plt.get_cmap("jet", n_eigenstates_lse)
cmap = plt.get_cmap("rainbow", n_eigenstates_lse)

# -------------------------------------------------------------------------------------------------
fig_dummy, ax_dummy = plt.subplots(dpi=100)

mat = np.arange(8).reshape((2, 4))

cs = ax_dummy.imshow(mat, cmap=cmap, vmin=0, vmax=n_eigenstates_lse-1)
# -------------------------------------------------------------------------------------------------

fig_conv_lse = plt.figure("figure_convergence_lse", figsize=(1.5*6, 1.5*4))

fig_conv_lse.subplots_adjust(left=0.175, right=0.9, bottom=0.2, top=0.9)

ax = fig_conv_lse.add_subplot(111)

ax.set_yscale('log')

ax.set_title('linear eigenstate computation')

for col in range(n_eigenstates_lse):
    plt.plot(vec_iter, matrix_res_batch[:, col], linewidth=1.5, linestyle='-', color=cmap(col))

# plt.plot(vec_iter, matrix_res_batch[:, 0], linewidth=1, linestyle='-', color='k')
# plt.plot(vec_iter, matrix_res_batch[:, 1], linewidth=1, linestyle='-', color='r')
# plt.plot(vec_iter, matrix_res_batch[:, 10], linewidth=1, linestyle='-', color='b')
# plt.plot(vec_iter, matrix_res_batch[:, 199], linewidth=1, linestyle='-', color='g')

ax.set_xlim(0, vec_iter[-1])
ax.set_ylim(1e-4, 1e0)

plt.xlabel(r'number of iterations', labelpad=12)
plt.ylabel(r'residual error', labelpad=12)

plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
# plt.grid(visible=False, which='minor', color='k', linestyle='-', linewidth=0.25)

cbar = fig_conv_lse.colorbar(cs, ax=ax)

cbar.set_ticks(np.array([0, 1, 2, 3, 4, 5, 6, 7])*7.0/8.0 + (7.0/8.0)/2.0)
cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7'])

cbar.ax.tick_params(length=8, pad=4, which="major")

fig_conv_lse.canvas.start_event_loop(0.001)

plt.draw()
"""
