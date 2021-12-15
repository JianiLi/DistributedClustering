import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

weight_loss = np.load("results/Weight_loss.npy")
weight_dist = np.load("results/Weight_dist.npy")

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
cmap = 'RdBu'

w_loss = axs[0].pcolormesh(np.average(weight_loss[:, :, :], axis=2), cmap=cmap)
w_dist = axs[1].pcolormesh(np.average(weight_dist, axis=2), cmap=cmap)

fig.tight_layout(pad=5.0)

axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel(r"Agent $l$", fontsize=15)
axs[0].set_ylabel(r"Agent $k$", fontsize=15)

axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlabel(r"Agent $l$", fontsize=15)
axs[1].set_ylabel(r"Agent $k$", fontsize=15)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
fig.colorbar(w_dist, cax=cbar_ax)

# plt.savefig('results/cluster_TL_weight_matrix.png', dpi=1000)

plt.show()
