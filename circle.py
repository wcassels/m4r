import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import jv, jn_zeros

import general_utils

num_boundary_nodes = 200
boundary_thetas = np.linspace(0, 2*np.pi-1e-1, num_boundary_nodes)

# unit disk boundary
boundaries = np.cos(boundary_thetas) + 1j*np.sin(boundary_thetas)


# plt.scatter(nodes.real, nodes.imag, s=3, c='red')
# plt.scatter(boundaries.real, boundaries.imag, s=3, c='green')
# plt.title(f"{num_points} domain nodes added")
# plt.grid()
# plt.show()

# have dirichlet 0 for now


time_step = 0.0005
diffusivity = 1
shape_param = 12



zero8 = jn_zeros(8, 2)[-1]
zero4 = jn_zeros(4,1)[0]
print(zero4)
A = B = 1
sol = lambda r , theta, t: jv(4, zero4 * r) * np.cos(4*theta) * np.exp(-zero4**2 * diffusivity * t)

# T = np.ones_like(nodes, dtype=np.float64)

# T0 = T.copy()

num_steps = 400

# cmin, cmax = np.min(T), np.max(T)
for num_domain_nodes in [50, 100, 200, 400, 600]:
    nodes = boundaries.copy()

    x_potential, y_potential = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    potentials = x_potential + 1j*y_potential

    # restrict potential nodes to those within the disk
    potentials = potentials[np.abs(potentials) < 1]

    # num_domain_nodes = 200

    for i in range(num_domain_nodes):
        print(i)
        ds = np.zeros_like(potentials, dtype=np.float64)

        for j in range(potentials.size):
            ds[j] = np.min(np.abs(nodes - potentials[j]))

        k = np.argmax(ds)
        nodes = np.append(nodes, potentials[k])
        potentials = np.delete(potentials, k)

    labels = np.concatenate((np.full(num_boundary_nodes, "D"), np.full(num_domain_nodes, None)))
    boundary_vals = np.concatenate((np.full(num_boundary_nodes, 0), np.full(num_domain_nodes, None)))
    deriv_lambdas = np.full(num_domain_nodes + num_boundary_nodes, None)

    neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)

    T = sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), 0)
    errs = np.zeros(num_steps+1)

    for t in range(1, num_steps+1):
        general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        errs[t] = np.mean(np.abs(T-sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t*time_step))) # avg abs errors
        # errs[t] = np.mean(np.abs(1-T/sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t*time_step)))
        # tests.append(np.median(T/T0))
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # rbf = ax[0].scatter(nodes.real, nodes.imag, c=T, s=3)
        # fig.colorbar(rbf, ax=ax[0])
        # rbf.set_clim(cmin, cmax)
        # ax[0].set_title(f"RBF Solution after {t * time_step:.3f} seconds\n{num_domain_nodes} domain nodes, Δt={time_step}, diff={diffusivity}")
        #
        # true = ax[1].scatter(nodes.real, nodes.imag, c=T-sol(nodes.real, nodes.imag, t*time_step), s=3)
        # fig.colorbar(true, ax=ax[1])
        # true.set_clim(cmin, cmax)
        # ax[1].set_title(f"Err")
        #
        # plt.show()

        # p = plt.scatter(nodes.real, nodes.imag, c=T, s=3)
        cmin, cmax = np.min(T), np.max(T)
        # if num_domain_nodes == 400:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(121, projection='3d')
        #     ax2 = fig.add_subplot(122, projection='3d')
        #     surf = ax.plot_trisurf(nodes.real, nodes.imag, T, vmin=cmin, vmax=cmax, cmap=cm.jet)
        #     surf2 = ax2.plot_trisurf(nodes.real, nodes.imag, sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t*time_step), vmin=cmin, vmax=cmax, cmap=cm.jet)
        #     # ax.set_zlim(cmin, cmax)
        #     # ax2.set_zlim(cmin, cmax)
        #     plt.colorbar(surf)
        #     # p.set_clim(cmin, cmax)
        #     ax.set_title(f"RBF Solution after {t*time_step:.3f} seconds")
        #     ax2.set_title("Analytical Solution")
        #     plt.show()
    plt.plot(errs, label=f"{num_domain_nodes} domain nodes")

# # tests2 = tests / tests[0]
# tests3 = -np.log(tests)
# tests4 = tests3 / (time_step * np.arange(len(tests3)))
# print(tests4)

# plt.plot(tests4)
# plt.plot(zero4**2 * time_step * np.arange(len(tests3)), label="true")
plt.legend()
plt.grid()
plt.title(f"RBF disk errors for eigenfunction initial condition\ndiff={diffusivity}, c={shape_param}, Δt={time_step}")
plt.show()
