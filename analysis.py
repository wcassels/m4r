"""
Functions to generate plots! (Will fill up when I copy across old code)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import rect_utils
import general_utils
import analyticals
import normal_derivs
import seaborn as sn
import normal_derivs

def first_test_general(time_step=1, num_time_steps=10000, plot_every=100, dx=0.05, c=10, method="Sarler"):
    x_min, x_max = 0.0, 0.6
    y_min, y_max = 0.0, 1.0
    shape_param = c
    grid_dist = dx

    # Diffusivity as described in the first test
    rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)

    boundary_conditions = {"North": ("R", (-750/k, 0)), "East": ("R", (-750/k, 0)),
                           "South": ("D", (100, None)), "West": ("N", (0, None))}
    avg_errs = []
    max_errs = []
    # grid_dists = [0.01, 0.025, 0.05, 0.1]
    Ns = [5, 6, 7, 8, 9, 10]

    for N in Ns:
    # for grid_dist in grid_dists:
        x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
        y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
        x, y = np.meshgrid(x_line, y_line)
        y = y[::-1]
        nodes = (x+y*1j).ravel()

        labels = np.full(nodes.size, None)
        # boundary_vals = np.full(nodes.size, None)
        boundary_vals = np.full(nodes.size, None)
        deriv_lambdas = np.full(nodes.size, None)


        # East boundary
        labels[np.isclose(nodes.real, 0.6)] = "R"
        for i in range(nodes.size):
            if np.isclose(nodes[i].real, 0.6):
                boundary_vals[i] = [-750/k, 0]

        deriv_lambdas[np.isclose(nodes.real, 0.6)] = lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="x", direction="-")

        # North boundary
        labels[np.isclose(nodes.imag, 1)] = "R"
        for i in range(nodes.size):
            if np.isclose(nodes[i].imag, 1):
                boundary_vals[i] = [-750/k, 0]
        deriv_lambdas[np.isclose(nodes.imag, 1)] = lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="y", direction="-")

        # West boundary
        labels[np.isclose(nodes.real, 0)] = "N"
        boundary_vals[np.isclose(nodes.real, 0)] = 0
        deriv_lambdas[np.isclose(nodes.real, 0)] = lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="x", direction="+")

        # South boundary
        labels[np.isclose(nodes.imag, 0)] = "D"
        boundary_vals[np.isclose(nodes.imag, 0)] = 100
        deriv_lambdas[np.isclose(nodes.imag, 0)] = lambda centre_pos, positions: normal_derivs.cartesian(centre_pos, positions, shape_param, axis="y", direction="+")

        # Initial condition according to
        # https://www.compassis.com/downloads/Manuals/Validation/Tdyn-ValTest7-Thermal_conductivity_in_a_solid.pdf
        # (seemingly omitted in Sarler paper)
        T = np.zeros_like(nodes, dtype=np.float64)

        # is_boundary = np.logical_or(np.logical_or(np.isclose(nodes.real, 0), np.isclose(nodes.real, 0.6)), np.logical_or(np.isclose(nodes.imag, 0), np.isclose(nodes.imag, 1)))
        # for i in range(nodes.size):
        #     pos = nodes[i]
        #     if not (np.isclose(pos.real, 0)  or np.isclose(pos.real, 0.6) or np.isclose(pos.imag, 0) or np.isclose(pos.imag, 1)):
        #         nodes[i] += np.random.uniform(-0.002, 0.002) + 1j * np.random.uniform(-0.002, 0.002)

        true_sol = analyticals.sarler_first(nodes.real, nodes.imag, k=k)
        neighbourhood_idx, weights = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N=5, method=method)
        rhs_vals = general_utils.filter_boundary_vals(boundary_vals, labels)

        # for t in range(1, num_time_steps+1):
        t = 0
        while True:
            t+=1
            T_old = T.copy()
            T = general_utils.step(T, weights, neighbourhood_idx, labels, rhs_vals, N=5, method=method)
            max_resid = np.max(np.abs(T - T_old))

            print(t)
            print("max residual: ", max_resid)
            print("err: ", np.linalg.norm(T - true_sol))
            if max_resid < 1e-6:
            # if t in [1, 1000]:
                print("avg err:", np.mean(np.abs(T - true_sol)))
                print("max err:", np.max(np.abs(T - true_sol)))
                avg_errs.append(np.mean(np.abs(T - true_sol)))
                max_errs.append(np.max(np.abs(T - true_sol)))
                break
                # fig = plt.figure()
                # ax = fig.gca(projection='3d')
                # surface = ax.plot_surface(x, y, T.reshape(*x.shape)-true_sol.reshape(*x.shape))
                # ax.set_xlabel('x')
                # ax.set_ylabel('y')
                # ax.view_init(elev=10, azim=110)
                # # title = ax.set_title(f'RBF solution at t = {t * time_step} seconds')
                # # title.set_position([.5, 1.025])
                # plt.tight_layout()
                # plt.savefig(f"report_figs/test1/steady_error_{method}.pdf", format="pdf")
                # plt.show()
    plt.semilogy(Ns, avg_errs)
    # plt.semilogy(grid_dists, max_errs)
    plt.grid()
    plt.xlabel("$N_\omega$")
    plt.ylabel("Error")
    plt.savefig("report_figs/test1/steady_N_error.pdf", format="pdf")
    np.save(f"data/test1/steady_N_avg", np.array(avg_errs))
    # np.save(f"data/test1/steady_dx_max", np.array(max_errs))

def second_test_general(time_step=1e-4, num_time_steps=1000, plot_every=100, dx=0.05, c=10, method="Sarler", N=5):
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    shape_param = c
    grid_dist = dx
    method = "Alternative"


    # Diffusivity as described in the first test
    diffusivity = 1.0

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1]
    nodes = (x+y*1j).ravel()

    labels = np.full(nodes.size, None)
    # boundary_vals = np.full(nodes.size, None)
    boundary_vals = np.full(nodes.size, None)
    deriv_lambdas = np.full(nodes.size, None)


    labels[nodes.real == 0] = "N"
    labels[nodes.real == 1] = "D"
    labels[nodes.imag == 1] = "D"
    labels[nodes.imag == 0] = "N"
    boundary_vals[nodes.real == 0] = 0
    boundary_vals[nodes.imag == 0] = 0
    boundary_vals[nodes.real == 1] = 0
    boundary_vals[nodes.imag == 1] = 0

    deriv_lambdas[nodes.real == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="+")
    deriv_lambdas[nodes.real == 1] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="-")
    deriv_lambdas[nodes.imag == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="+")
    deriv_lambdas[nodes.real == 1] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="-")

    # for method in ["Sarler", "Alternative"]:
    for shape_param in [1, 4, 16, 64, 128]:

        T = np.ones_like(nodes, dtype=np.float64)

        neighbourhood_idx, weights = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N=N, method=method)
        rhs_vals = general_utils.filter_boundary_vals(boundary_vals, labels)
        # for t in range(1, num_time_steps+1):
        errs = [0]
        t = 0
        # while True:
        for t in range(1, num_time_steps+1):
            # t+=1
            T_old = T.copy()
            T = general_utils.step(T, weights, neighbourhood_idx, labels, rhs_vals, N=5, method=method)
            max_resid = np.max(np.abs(T - T_old))

            print(shape_param, t, "max residual: ", max_resid)
            trunc_sol = analyticals.sarler_second(nodes.real, nodes.imag, t*time_step, diffusivity, trunc=50)
            # print(t)
            print(T.shape)
            print(trunc_sol.shape)
            errs.append(np.mean(np.abs(T - trunc_sol)))
            print("err: ", np.mean(np.abs((T - trunc_sol))))
            # if max_resid < 1e-3:
            # if t in [1, 10]:
                # fig = plt.figure()
                # ax = fig.gca(projection='3d')
                # surface = ax.plot_surface(x, y, T.reshape(*x.shape))
                # ax.set_xlabel('x')
                # ax.set_ylabel('y')
                # ax.view_init(elev=10, azim=110)
                # # title = ax.set_title(f'RBF solution at t = {t * time_step} seconds')
                # # title.set_position([.5, 1.025])
                # # plt.tight_layout()
                # ax.tick_params(axis='z', pad=15)
                # plt.savefig(f"report_figs/test2/{t}_{method}.pdf", format="pdf")
                # plt.show()
        # np.save(f"data/test2/errs_{method}_{N}", np.array(errs))
        plt.semilogy(range(1, len(errs)+1), errs, label=f"c={shape_param}")
    plt.xlabel("Number of time steps")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_figs/test2/shape_comp.pdf")

                # break
