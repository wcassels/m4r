import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import general_utils
import normal_derivs

num_boundary_nodes = 100
boundary_thetas = np.linspace(0, 2*np.pi, num_boundary_nodes+1)[:-1]

# unit disk boundary
boundaries = np.cos(boundary_thetas) + 1j*np.sin(boundary_thetas)
nodes = boundaries.copy()

# # inner disk
# inner_ratio = 0.95
# nodes = np.append(nodes, inner_ratio * boundaries)

time_step = 0.001
diffusivity = 1
shape_param = 16

num_steps = 600

num_domain_nodes = 1000


x_potential, y_potential = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
potentials = x_potential + 1j*y_potential

# restrict potential nodes to those within the inner disk
potentials = potentials[np.abs(potentials) < 1]#*0.9]

num_domain_nodes = 200

for i in range(num_domain_nodes):
    print(i)
    ds = np.zeros_like(potentials, dtype=np.float64)

    for j in range(potentials.size):
        ds[j] = np.min(np.abs(nodes - potentials[j]))

    k = np.argmax(ds)
    nodes = np.append(nodes, potentials[k])
    potentials = np.delete(potentials, k)

# np.save(f"node_positions/disk/{num_boundary_nodes}boundary{num_domain_nodes}domain", nodes)
labels = np.concatenate((np.full(num_boundary_nodes, "N"), np.full(num_domain_nodes, None)))  # due to inner diks
boundary_vals = np.concatenate((np.full(num_boundary_nodes, 0), np.full(num_domain_nodes, None)))
deriv_lambdas = np.concatenate((np.full(num_boundary_nodes, lambda centre, pos: normal_derivs.radial(centre, pos, shape_param)), np.full(num_domain_nodes, None)))

###

## UNIFORM POLAR NODES INCLUDING ON THE BOUNARY
# num_r, num_theta = 9, 25 # 11, 31 is stable but still doesnt look like 0 derivative
# r_pot, theta_pot = np.linspace(0,1,num_r+1)[1:], np.linspace(0, 2*np.pi, num_theta+1)[:-1]
# r, t = np.meshgrid(r_pot, theta_pot)
# r, t = r.ravel(), t.ravel()
# num_domain_nodes = num_theta * (num_r - 1) + 1
# num_boundary_nodes = num_theta
# # nodes = np.concatenate((nodes, r*np.cos(t)+1j*r*np.sin(t)))
# nodes = r*np.cos(t) + 1j* r*np.sin(t)
# r = np.append(r, 0)
# nodes = np.append(nodes, 0+0j)
# plt.scatter(nodes.real, nodes.imag)
# plt.show()
# num_domain_nodes = nodes.size

# labels = np.concatenate((np.full(num_boundary_nodes, "N"), np.full(num_domain_nodes, None)))
# labels = np.full(num_domain_nodes + num_boundary_nodes, None)
# boundary_vals = np.full(num_domain_nodes + num_boundary_nodes, None)
# deriv_lambdas = np.full(num_domain_nodes + num_boundary_nodes, None)
# labels[r==1] = "N"
# boundary_vals[r==1] = 0
# deriv_lambdas[r==1] = lambda centre, pos: normal_derivs.radial(centre, pos, shape_param)

##


neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)

# T = sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), 0)
# T = np.ones_like(nodes, dtype=np.float64)
T = np.exp(-(np.abs(nodes))**2)
# T = np.cos(np.abs(nodes))
# T = np.abs(nodes)
# T = np.ones_like(nodes, dtype=np.float64)

cmin, cmax = np.min(T), np.max(T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(nodes.real, nodes.imag, T, vmin=cmin, vmax=cmax, cmap=cm.jet)

# ax.set_zlim(cmin, cmax)
# ax2.set_zlim(cmin, cmax)
plt.colorbar(surf)
# p.set_clim(cmin, cmax)
ax.set_title(f"Initial condition")

plt.show()

for t in range(1, num_steps+1):

    general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
    print(T[-1])
    # errs[t] = np.mean(np.abs(T-sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t*time_step))) # avg abs errors
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(nodes.real, nodes.imag, T, vmin=cmin, vmax=cmax, cmap=cm.jet)

    # ax.set_zlim(cmin, cmax)
    # ax2.set_zlim(cmin, cmax)
    plt.colorbar(surf)
    # p.set_clim(cmin, cmax)
    ax.set_title(f"RBF Solution after {t*time_step:.3f} seconds")

    plt.show()
    # plt.plot(errs, label=f"{num_domain_nodes} domain nodes")

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
