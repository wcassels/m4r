import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import general_utils
import normal_derivs
import node_utils

shape_param = 40
time_step = 0.0001
diffusivity = 1
num_steps = 10000

## OUTER RECTANGLE
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
num_ns_nodes, num_ew_nodes = 50, 50
condition_dict = {"North": "D",
                  "East": "D",
                  "South": "D",
                  "West": "D"}

value_dict = {"North": 0,
              "East": 0,
              "South": 0,
              "West": 0}

deriv_dict = {"North": None,
              "East": None,
              "South": None,
              "West": None}

boundary_nodes, labels, boundary_vals, deriv_lambdas = node_utils.make_rectangle(x_min, x_max, y_min, y_max, num_ns_nodes, num_ew_nodes, condition_dict, value_dict, deriv_dict)
domain_conditions = [lambda p: x_min * .99 < p.real, lambda p: p.real < x_max * .99, lambda p: y_min < p.imag * .99, lambda p: p.imag < y_max * .99]

## INNER RECTANGLE
x_min2, x_max2 = -0.2, 0.2
y_min2, y_max2 = -0.2, 0.2
num_ns_nodes2, num_ew_nodes2 = 10, 10

# only thing that is changing is the value of the dirichlet condition
value_dict = {"North": 1,
              "East": 1,
              "South": 1,
              "West": 1}

boundary_nodes2, labels2, boundary_vals2, deriv_lambdas2 = node_utils.make_rectangle(x_min2, x_max2, y_min2, y_max2, num_ns_nodes2, num_ew_nodes2, condition_dict, value_dict, deriv_dict)

boundary_nodes = np.concatenate((boundary_nodes, boundary_nodes2))
labels = np.concatenate((labels, labels2))
boundary_vals = np.concatenate((boundary_vals, boundary_vals2))
deriv_lambdas = np.concatenate((deriv_lambdas, deriv_lambdas2))

num_domain_nodes = 1000

domain_conditions2 = [lambda p: np.logical_or(np.logical_or(p.real < x_min2 * 1.01, p.real > x_max2 * 1.01), np.logical_or(p.imag < y_min2 * 1.01, p.imag > y_max2 * 1.01))]
domain_conditions += domain_conditions2

plt.scatter(boundary_nodes.real, boundary_nodes.imag, s=3)
plt.show()

# cut outs to plot NaNs
domain_nodes, cut_outs = node_utils.fill_domain(boundary_nodes, domain_conditions, num_domain_nodes)

nodes = np.concatenate((boundary_nodes, domain_nodes))
labels = np.concatenate((labels, np.full(num_domain_nodes, None)))
boundary_vals = np.concatenate((boundary_vals, np.full(num_domain_nodes, None)))
deriv_lambdas = np.concatenate((deriv_lambdas, np.full(num_domain_nodes, None)))

plt.scatter(nodes.real, nodes.imag, s=3)
plt.show()

T = np.ones_like(nodes, dtype=np.float64)

neighbourhood_idx, weights, boundary_flags = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N=8, N_boundary=5, method="Alternative")

plotting_nodes = np.concatenate((nodes, cut_outs))
t = 0

while True:
    t += 1
    T_old = T.copy()
    general_utils.step(T, weights, neighbourhood_idx, labels, general_utils.filter_boundary_vals(boundary_vals, labels), N=8, N_boundary=5, boundary_flags=boundary_flags, method="Alternative")

    max_resid = np.max(np.abs(T-T_old))
    print("max residual: ", max_resid)

    if t % 1000 == 0 or t == 1 or max_resid <= 1e-6:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_trisurf(nodes.real, nodes.imag, T)#, vmin=cmin, vmax=cmax, cmap=cm.jet)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.savefig(f"report_figs/ex2/{t}.pdf")

        if max_resid <= 1e-6:
            break
