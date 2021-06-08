import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import general_utils
import normal_derivs
import node_utils

# outer boundary is a rectangle
# x_min, x_max = -2.0, 2.0
# y_min, y_max = -3.0, 3.0
# outer_boundary = lambda p: x_min <= p.real <= x_max and y_min <= p.imag <= y_max
#
# all_boundary_lambdas = []
#
# all_boundary_nodes

shape_param = 40
time_step = 0.0001
diffusivity = .1
num_steps = 10000

## DEFINE RECTANGLE
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
num_ns_nodes, num_ew_nodes = 50, 50
condition_dict = {"North": "N",
                  "East": "N",
                  "South": "N",
                  "West": "N"}

value_dict = {"North": 0,
              "East": 0,
              "South": 0,
              "West": 0}

deriv_dict = {"North": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="y", direction="-"),
              "East": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="x", direction="-"),
              "South": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="y", direction="+"),
              "West": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="x", direction="+")}

boundary_nodes, labels, boundary_vals, deriv_lambdas = node_utils.make_rectangle(x_min, x_max, y_min, y_max, num_ns_nodes, num_ew_nodes, condition_dict, value_dict, deriv_dict)

num_domain_nodes = 1000

domain_conditions = [lambda p: x_min * .99 < p.real, lambda p: p.real < x_max * .99, lambda p: y_min < p.imag * .99, lambda p: p.imag < y_max * .99]

# add two circles
circle_centre, circle_rad = -0.6, 0.2
domain_conditions.append(lambda p: np.abs(p-circle_centre) > circle_rad)
circle_nodes, circle_labels, circle_boundary_vals, circle_deriv_lambdas = node_utils.make_circle(circle_centre, circle_rad, 15, "N", 0, lambda centre_pos, positions: normal_derivs.radial(centre_pos, positions, shape_param, direction="outwards", circ_centre=circle_centre))

circle_centre2, circle_rad2 = 0.6, 0.2
domain_conditions.append(lambda p: np.abs(p-circle_centre2) > circle_rad2)
circle_nodes2, circle_labels2, circle_boundary_vals2, circle_deriv_lambdas2 = node_utils.make_circle(circle_centre2, circle_rad2, 15, "N", 0, lambda centre_pos, positions: normal_derivs.radial(centre_pos, positions, shape_param, direction="outwards", circ_centre=circle_centre2))

# domain_conditions.append(lambda p: np.logical_or(p.real>0.4, p.real<-0.1))
# domain_conditions.append(lambda p: np.logical_or(p.imag>0.2, p.imag<-0.5))
# condition_dict = {"North": "D",
#                   "East": "D",
#                   "South": "D",
#                   "West": "D"}
#
# value_dict = {"North": 0,
#               "East": 0,
#               "South": 0,
#               "West": 0}
#
# deriv_dict = {"North": None,
#               "East": None,
#               "South": None,
#               "West": None}
# circle_nodes, circle_labels, circle_boundary_vals, circle_deriv_lambdas = node_utils.make_rectangle(-0.1, 0.4, -0.5, 0.2, 5, 5, condition_dict, value_dict, deriv_dict)


boundary_nodes = np.concatenate((boundary_nodes, circle_nodes, circle_nodes2))
labels = np.concatenate((labels, circle_labels, circle_labels2))
boundary_vals = np.concatenate((boundary_vals, circle_boundary_vals, circle_boundary_vals2))
deriv_lambdas = np.concatenate((deriv_lambdas, circle_deriv_lambdas, circle_deriv_lambdas2))

plt.scatter(boundary_nodes.real, boundary_nodes.imag, s=3)
plt.show()

# cut outs to plot NaNs
domain_nodes, cut_outs = node_utils.fill_domain(boundary_nodes, domain_conditions, num_domain_nodes, autosave=25)

nodes = np.concatenate((boundary_nodes, domain_nodes))
labels = np.concatenate((labels, np.full(num_domain_nodes, None)))
boundary_vals = np.concatenate((boundary_vals, np.full(num_domain_nodes, None)))
deriv_lambdas = np.concatenate((deriv_lambdas, np.full(num_domain_nodes, None)))

plt.scatter(nodes.real, nodes.imag, s=3)
plt.show()

#T = np.ones_like(nodes, dtype=np.float64)#
T = np.exp(-25 * (np.abs(nodes-0.5j)**2))

# neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)
neighbourhood_idx, weights, boundary_flags = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N=7, N_boundary=5, method="Alternative")

# plotting_nodes = np.concatenate((nodes, cut_outs))
t = 0
# for t in range(1, num_steps+1):
while True:
    t += 1
    T_old = T.copy()
    # general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
    general_utils.step(T, weights, neighbourhood_idx, labels, general_utils.filter_boundary_vals(boundary_vals, labels), N=7, N_boundary=5, boundary_flags=boundary_flags, method="Alternative")

    # cmin, cmax = np.min(T), np.max(T)
    # plotting_T = np.concatenate((T, np.full(cut_outs.size, np.nan)))
    max_resid = np.max(np.abs(T-T_old))
    print("max residual: ", max_resid)


    # ax.set_zlim(cmin, cmax)
    # ax2.set_zlim(cmin, cmax)
    # plt.colorbar(surf)
    # p.set_clim(cmin, cmax)
    # plt.title(f"RBF Solution after {t*time_step:.3f} seconds")
    # fig.colorbar(sc)
    if t % 1000 == 0 or t == 1 or max_resid <= 1e-6:
    # if True:
    # if True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # sc = plt.scatter(nodes.real, nodes.imag, c=T, s=3)
        surf = ax.plot_trisurf(nodes.real, nodes.imag, T)#, vmin=cmin, vmax=cmax, cmap=cm.jet)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlim(0,1)
        # plt.show()
        plt.savefig(f"report_figs/ex3/{t}.pdf", format="pdf")

        if max_resid <= 1e-6:
            break

# plt.scatter(boundary_nodes.real, boundary_nodes.imag, s=3)
# plt.scatter(domain_nodes.real, domain_nodes.imag, s=3)
# plt.show()
