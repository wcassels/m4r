import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import general_utils
import normal_derivs
import node_utils

## OK SO
# outer boundary is a rectangle
# x_min, x_max = -2.0, 2.0
# y_min, y_max = -3.0, 3.0
# outer_boundary = lambda p: x_min <= p.real <= x_max and y_min <= p.imag <= y_max
#
# all_boundary_lambdas = []
#
# all_boundary_nodes

shape_param = 12
time_step = 0.0001
diffusivity = 1
num_steps = 10000

## DEFINE RECTANGLE
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
num_ns_nodes, num_ew_nodes = 20, 20
condition_dict = {"North": "D",
                  "East": "D",
                  "South": "N",
                  "West": "N"}

value_dict = {"North": 0,
              "East": 0,
              "South": 0,
              "West": 0}

deriv_dict = {"North": None,
              "East": None,
              "South": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="y", direction="+"),
              "West": lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="x", direction="+")}

boundary_nodes, labels, boundary_vals, deriv_lambdas = node_utils.make_rectangle(x_min, x_max, y_min, y_max, num_ns_nodes, num_ew_nodes, condition_dict, value_dict, deriv_dict)
# inner_rect_nodes, _, _, _ = node_utils.make_rectangle(-0.9, 0.9, -0.9, 0.9, num_ns_nodes-1, num_ew_nodes-1, condition_dict, value_dict, deriv_dict)
inner_rect_nodes = np.array([], dtype=np.complex128)
num_inner_rect_nodes = inner_rect_nodes.size


num_domain_nodes = 1000

domain_conditions = [lambda p: x_min < p.real, lambda p: p.real < x_max, lambda p: y_min < p.imag, lambda p: p.imag < y_max]

# put circle in middle
circle_centre, circle_rad = 0.0, 0.4
domain_conditions.append(lambda p: np.abs(p-circle_centre) > circle_rad)
circle_nodes, circle_labels, circle_boundary_vals, circle_deriv_lambdas = node_utils.make_circle(circle_centre, circle_rad, 30, "D", 2, lambda centre_pos, positions: normal_derivs.radial(centre_pos, positions, shape_param, direction="outwards"))
# calling them circle_etc because i cant be bothered to change the names...
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


boundary_nodes = np.concatenate((boundary_nodes, circle_nodes, inner_rect_nodes))
labels = np.concatenate((labels, circle_labels))
boundary_vals = np.concatenate((boundary_vals, circle_boundary_vals))
deriv_lambdas = np.concatenate((deriv_lambdas, circle_deriv_lambdas))

# cut outs to plot NaNs
domain_nodes, cut_outs = node_utils.fill_domain(boundary_nodes, domain_conditions, num_domain_nodes)

nodes = np.concatenate((boundary_nodes, domain_nodes))
labels = np.concatenate((labels, np.full(num_domain_nodes+num_inner_rect_nodes, None)))
boundary_vals = np.concatenate((boundary_vals, np.full(num_domain_nodes+num_inner_rect_nodes, None)))
deriv_lambdas = np.concatenate((deriv_lambdas, np.full(num_domain_nodes+num_inner_rect_nodes, None)))

plt.scatter(nodes.real, nodes.imag, s=3)
plt.show()

T = np.ones_like(nodes, dtype=np.float64)

neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)

plotting_nodes = np.concatenate((nodes, cut_outs))

for t in range(1, num_steps+1):
    general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)

    cmin, cmax = np.min(T), np.max(T)
    plotting_T = np.concatenate((T, np.full(cut_outs.size, np.nan)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # sc = plt.scatter(nodes.real, nodes.imag, c=T, s=3)
    surf = ax.plot_trisurf(nodes.real, nodes.imag, T, vmin=cmin, vmax=cmax, cmap=cm.jet)

    # ax.set_zlim(cmin, cmax)
    # ax2.set_zlim(cmin, cmax)
    plt.colorbar(surf)
    # p.set_clim(cmin, cmax)
    plt.title(f"RBF Solution after {t*time_step:.3f} seconds")
    # fig.colorbar(sc)

    plt.show()

plt.scatter(boundary_nodes.real, boundary_nodes.imag, s=3)
plt.scatter(domain_nodes.real, domain_nodes.imag, s=3)
plt.show()
