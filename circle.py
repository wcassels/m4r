import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import jv, jn_zeros

import general_utils
import node_utils

num_boundary_nodes = 75
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
shape_param = 20
method = "Alternative"

zero8 = jn_zeros(8, 2)[-1]
zero4 = jn_zeros(4,1)[0]
print(zero4)
A = B = 1
sol = lambda r , theta, t: jv(4, zero4 * r) * np.cos(4*theta) * np.exp(-zero4**2 * diffusivity * t)

# T = np.ones_like(nodes, dtype=np.float64)

# T0 = T.copy()

num_steps = 1000

# cmin, cmax = np.min(T), np.max(T)
# for num_domain_nodes in [50, 100, 200, 400, 600]:
Ns = [3, 4, 5, 7, 9]
# for num_domain_nodes in [200]:
for N in Ns:
    nodes = boundaries.copy()

    x_potential, y_potential = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    potentials = x_potential + 1j*y_potential

    # restrict potential nodes to those within the disk
    potentials = potentials[np.abs(potentials) < 1]

    num_domain_nodes = 700
    nodes, _ = node_utils.fill_domain(boundaries, [lambda p: np.abs(p)<1], num_domain_nodes)
    nodes = np.concatenate((boundaries, nodes))

    # for i in range(num_domain_nodes):
    #     print(i)
    #     ds = np.zeros_like(potentials, dtype=np.float64)
    #
    #     for j in range(potentials.size):
    #         ds[j] = np.min(np.abs(nodes - potentials[j]))
    #
    #     k = np.argmax(ds)
    #     nodes = np.append(nodes, potentials[k])
    #     potentials = np.delete(potentials, k)

    labels = np.concatenate((np.full(num_boundary_nodes, "D"), np.full(num_domain_nodes, None)))
    boundary_vals = np.concatenate((np.full(num_boundary_nodes, 0), np.full(num_domain_nodes, None)))
    deriv_lambdas = np.full(num_domain_nodes + num_boundary_nodes, None)

    neighbourhood_idx, update_weights = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N=N, method=method)

    T = sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), 0)
    errs = np.zeros(num_steps+1, dtype=np.float64)

    # sc = plt.scatter(nodes.real, nodes.imag, s=3)
    # plt.show()

    for t in range(1, num_steps+1):
        T = general_utils.step(T, update_weights, neighbourhood_idx, labels, general_utils.filter_boundary_vals(boundary_vals, labels), method=method)
        print(T.shape)
        print(t, "done")
        errs[t] = np.mean(np.abs(T-sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t*time_step))) # avg abs errors
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")

        # surf = ax.plot_trisurf(nodes.real, nodes.imag, T)
        # surf = ax.plot_trisurf(nodes.real, nodes.imag, sol(np.abs(nodes), np.arctan(nodes.imag/nodes.real), t * time_step))

        # plt.show()
    plt.semilogy(range(num_steps+1), errs, label=f"N={N}")

plt.legend()
plt.grid()
plt.xlabel("Number of time steps")
plt.ylabel("Error")
# plt.title(f"RBF disk errors for eigenfunction initial condition\ndiff={diffusivity}, c={shape_param}, Δt={time_step}")
# plt.show()
plt.savefig("report_figs/disk_d/Ns.pdf", format="pdf")
