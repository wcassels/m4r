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


def first_test(time_step=1.0e-4, num_steps=50, plot_every=20, shape_param=4):
    x_min, x_max = 0.0, 0.6
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the first test
    rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)

    # test 1
    boundary_conditions = {"North": ("Robin", (-750/k, 0)), "East": ("Robin", (-750/k, 0)),
                           "South": ("Dirichlet", (100, None)), "West": ("Neumann", (0, None))}

    # second grid layout
    grid_dist = 0.02

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1]

    # Initial condition according to
    # https://www.compassis.com/downloads/Manuals/Validation/Tdyn-ValTest7-Thermal_conductivity_in_a_solid.pdf
    # (seemingly omitted in Sarler paper)
    T = np.zeros_like(x, dtype=np.float64)

    # lazily chose points in any order because with uniform x/y grid_dist this doesn't matter,
    # but if testing cases where \Delta x != \Delta y, this requires more caution
    update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, shape_param)

    for t in range(1, num_steps+1):
        rect_utils.step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=rect_utils.unif_boundary)

        if (t % plot_every) == 0:
            print(f"NAFEMs convergence val: {T[-11,-1]}")
            # plt.imshow()
            # sn.heatmap(T)
            # plt.show()
            fig = plt.figure(figsize=(12,6))
            ax = fig.gca(projection='3d')
            surface = ax.plot_surface(x, y, T)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.view_init(elev=10, azim=110)
            ax.set_title(f'RBF solution')
            plt.show()

    return


def first_test_mod(time_step=1.0e-1, num_steps=50, plot_every=20, shape_param=4, grid_dist=0.05, N=5, method="Sarler", jumps=1):
    # formulation of first sarler grid on general domain to compare modified
    # scheme
    # second test
    x_min, x_max = 0.0, 0.6
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the first test
    rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)
    h = 750

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)

    nodes = x.ravel() + 1j*y.ravel()
    labels = np.full(nodes.size, None)
    boundary_vals = np.full(nodes.size, None)
    deriv_lambdas = np.full(nodes.size, None)

    # print((np.abs(nodes.real - 0.6)<0.001).any())


    # labels[np.abs(nodes.real - 0.6)<0.001] = "R" # east
    # labels[nodes.real == 0] = "N" # west
    # labels[nodes.imag == 1] = "R" # north
    # labels[nodes.imag == 0] = "D" # south

    # have to loop and do the robin nodes manually
    for i in range(nodes.size):
        node = nodes[i]

        if node.imag == 0:
            labels[i] = "D"
            boundary_vals[i] = 100
        elif node.real == 0:
            boundary_vals[i] = 0
            labels[i] = "N"
        elif node.imag == 1 or abs(node.real - 0.6) < 0.001:
            boundary_vals[i] = [-h/k, 0]
            labels[i] = "R"

    # boundary_vals[nodes.real == 0] = 0 # west
    # # boundary_vals[np.abs(nodes.real - 0.6)<0.001] = [-h/k, 0] # east
    # boundary_vals[nodes.imag == 0] = 100 # south
    # boundary_vals[nodes.imag == 1] = [-h/k, 0] # north



    deriv_lambdas[nodes.real == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="+")
    deriv_lambdas[np.abs(nodes.real - 0.6)<0.001] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="-")
    deriv_lambdas[nodes.imag == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="+")
    deriv_lambdas[nodes.imag == 1] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="-")

    # neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)
    neighbourhood_idx, weights = general_utils.setup(nodes, labels, boundary_vals, deriv_lambdas, time_step, diffusivity, shape_param, N, method=method)
    # neighbourhood_idx, update_info = general_utils.alt_setup_two(nodes, labels, deriv_lambdas, time_step, diffusivity, shape_param)

    T = np.zeros_like(nodes, dtype=np.float64)

    errs = [0]
    t = 0

    rhs_vals = general_utils.filter_boundary_vals(boundary_vals, labels)

    for t in range(1, num_steps+1):

        # T_old = T.copy()
        # general_utils.generalised_everywhere_step(T, update_info, neighbourhood_idx, labels, boundary_vals)
        # general_utils.modified_implicit_step(T, update_info, neighbourhood_idx, labels, boundary_vals)
        # general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        T = general_utils.step(T, weights, neighbourhood_idx, labels, rhs_vals, method=method, jumps=jumps)


        if (t % plot_every) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, T.reshape((*x.shape)))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        # trunc_sol = analyticals.sarler_second(x, y, t*time_step, diffusivity, trunc=trunc)
        #
        # errs.append(np.mean(np.abs(T.reshape((41,41))-trunc_sol)))
        # print(f"t={t*time_step:.5f}\tMax abs change: {np.max(np.abs(T-T_old))}")
        # if np.max(np.abs(T-T_old)) < convergence_crit:
        #     break

    # plt.plot(range(len(errs)), errs)
    # plt.xlabel("Time steps")
    # plt.ylabel("Error")
    # plt.title(f"Average solution error (Δt={time_step})")
    # plt.grid()
    # plt.show()

    return


def first_test_errs(time_step=0.1, convergence_tol=1e-6):
    """
    Runs until convergence, and produces an avg abs error plot
    """
    x_min, x_max = 0.0, 0.6
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the first test
    rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)

    # test 1
    boundary_conditions = {"North": ("Robin", (-750/k, 0)), "East": ("Robin", (-750/k, 0)),
                           "South": ("Dirichlet", (100, None)), "West": ("Neumann", (0, None))}

    # second grid layout
    grid_dist = 0.02

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1]

    true_sol = analyticals.sarler_first(x, y, k=k)

    t = 0

    cs = 4*np.arange(1, 11)
    errs = np.zeros_like(cs, dtype=np.float64)
    for i, c in enumerate(cs):
        # lazily chose points in any order because with uniform x/y grid_dist this doesn't matter,
        # but if testing cases where \Delta x != \Delta y, this requires more caution
        update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, c)
        T = np.zeros_like(x, dtype=np.float64)
        t = 0


        while True:
            t += 1
            T_old = T.copy()
            rect_utils.step(T, update_weights, grid_dist, c, boundary_conditions, boundary_method=rect_utils.unif_boundary)
            T[0,0] = true_sol[0,0]
            T[0,-1] = true_sol[0,-1]
            T[-1,0] = true_sol[-1,0]
            T[-1,-1] = true_sol[-1,-1]

            print(f"c={c}\tt={t*time_step:.1f}\tMax abs change: {np.max(np.abs(T-T_old))}")

            if  np.max(np.abs(T-T_old)) < convergence_tol:
                errs[i] = np.mean(np.abs(T-true_sol))
                break

    plt.semilogy(cs, errs)
    plt.grid()
    plt.xlabel("Shape parameter")
    plt.ylabel("Avg error")
    plt.title(f"Uniform configuration: average abs error at convergence\nΔt={time_step}, Δx={grid_dist}, tol={convergence_tol}")
    plt.show()
    return


def second_test_comparison(time_step=1.0e-4, num_steps=50, plot_every=20, trunc=50, shape_param=4):
    """
    Allows for comparison of the RBF and analytical solutions as we step forward in time.
    """
    # second test
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the second test
    rho, param_c, k = 1.0, 1.0, 1.0
    diffusivity = k / (rho * param_c)


    # test 2
    boundary_conditions = {"North": ("Dirichlet", (0, None)), "East": ("Dirichlet", (0, None)),
                            "South": ("Neumann", (0, None)), "West": ("Neumann", (0, None))}

    grid_dist = 0.025

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1] # by default meshgrid returns a y that increases going down the matrix
                # ie. the opposite of the intuitive picture - need to reverse this
                # decide whether to do this here or just remember how it works and change the
                # boundary conditions implementation

    # Initial condition
    T = np.ones_like(x, dtype=np.float64)

    # Get simplified update weights - only works for uniform configuration!
    # update_weights = rect_utils.get_update_weights(Phi, diffusivity, time_step, grid_dist, shape_param)
    update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, shape_param)

    for t in range(1, num_steps+1):
        rect_utils.step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=rect_utils.unif_boundary)
        # Corner values are not computed. They also have no influence on future calculations

        if (t % plot_every) == 0:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            surface = ax.plot_surface(x, y, T)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'RBF solution')

            trunc_sol = analyticals.sarler_second(x, y, t*time_step, diffusivity, trunc=trunc)
            print(f"t={t*time_step:.3f}\tMean abs err: {np.mean(np.abs(T-trunc_sol))}")
            ax = fig.add_subplot(1, 2, 2, projection="3d")
            surface = ax.plot_surface(x, y, trunc_sol)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Analytical Solution (Truncated)')

            plt.suptitle(f"Comparison after {t} time steps (Δt={time_step})")
            plt.tight_layout()
            plt.show()

    return

def second_test_mod(time_step, trunc=50, shape_param=12, convergence_crit=1e-6):
    # formulation of second sarler grid on general domain to compare modified
    # scheme
    # second test
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the second test
    rho, param_c, k = 1.0, 1.0, 1.0
    diffusivity = k / (rho * param_c)

    grid_dist = 0.025

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)

    nodes = x.ravel() + 1j*y.ravel()
    labels = np.full(nodes.size, None)
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

    neighbourhood_idx, update_info = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param)
    # neighbourhood_idx, update_info = general_utils.alt_setup_two(nodes, labels, deriv_lambdas, time_step, diffusivity, shape_param)

    T = np.ones_like(nodes, dtype=np.float64)

    errs = [0]
    t = 0
    while True:
        t += 1

        T_old = T.copy()
        # general_utils.generalised_everywhere_step(T, update_info, neighbourhood_idx, labels, boundary_vals)
        # general_utils.modified_implicit_step(T, update_info, neighbourhood_idx, labels, boundary_vals)
        general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        trunc_sol = analyticals.sarler_second(x, y, t*time_step, diffusivity, trunc=trunc)

        errs.append(np.mean(np.abs(T.reshape((41,41))-trunc_sol)))
        print(f"t={t*time_step:.5f}\tMax abs change: {np.max(np.abs(T-T_old))}")
        if np.max(np.abs(T-T_old)) < convergence_crit:
            break

    plt.plot(range(len(errs)), errs)
    plt.xlabel("Time steps")
    plt.ylabel("Error")
    plt.title(f"Average solution error (Δt={time_step})")
    plt.grid()
    plt.show()

    return


def second_test_avg_errs(time_step, trunc=50, shape_param=12, convergence_crit=1e-6):
    # second test
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the second test
    rho, param_c, k = 1.0, 1.0, 1.0
    diffusivity = k / (rho * param_c)


    # test 2
    boundary_conditions = {"North": ("Dirichlet", (0, None)), "East": ("Dirichlet", (0, None)),
                            "South": ("Neumann", (0, None)), "West": ("Neumann", (0, None))}

    grid_dist = 0.025

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1] # by default meshgrid returns a y that increases going down the matrix
                # ie. the opposite of the intuitive picture - need to reverse this
                # decide whether to do this here or just remember how it works and change the
                # boundary conditions implementation

    # Initial condition
    T = np.ones_like(x, dtype=np.float64)

    # Get domain update weights
    update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, shape_param)

    errs = [0]
    t = 0
    while True:
        t += 1

        T_old = T.copy()
        rect_utils.step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=rect_utils.unif_boundary)

        trunc_sol = analyticals.sarler_second(x, y, t*time_step, diffusivity, trunc=trunc)

        # Correct corners - this has no effect on RBF solution marching
        T[0,0], T[0,-1], T[-1,0], T[-1,-1] = trunc_sol[0,0], trunc_sol[0,-1], trunc_sol[-1,0], trunc_sol[-1,-1]

        errs.append(np.mean(np.abs(T-trunc_sol)))
        print(f"t={t*time_step:.5f}\tMax abs change: {np.max(np.abs(T-T_old))}")
        if np.max(np.abs(T-T_old)) < convergence_crit:
            break

    plt.plot(range(len(errs)), errs)
    plt.xlabel("Time steps")
    plt.ylabel("Error")
    plt.title(f"Average solution error (Δt={time_step})")
    plt.grid()
    plt.show()

    return


def first_test_NAFEMs_convergence(time_step=1, convergence_crit=1.0e-6, diff=None, max_steps=10000):
    """
    Plots evolution of value at (0.6, 0.2), referenced in the Sarler paper.
    Continues until the value has convergenced according to provided convergence
    criterion
    """
    # Rounded 8 digit analytical solution at equilibrium
    LIMIT = 18.253756

    x_min, x_max = 0.0, 0.6
    y_min, y_max = 0.0, 1.0

    # Diffusivity as described in the first test
    rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)

    # test 1
    boundary_conditions = {"North": ("Robin", (-750/k, 0)), "East": ("Robin", (-750/k, 0)),
                           "South": ("Dirichlet", (100, None)), "West": ("Neumann", (0, None))}

    # second grid layout
    grid_dist = 0.02

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1]

    # Initial condition according to
    # https://www.compassis.com/downloads/Manuals/Validation/Tdyn-ValTest7-Thermal_conductivity_in_a_solid.pdf
    # (omitted in Sarler paper)
    T = np.zeros_like(x, dtype=np.float64)

    cs = np.array([4, 8, 16, 32])
    converged_vals = np.zeros_like(cs, dtype=np.float64)
    convergence_times = np.zeros_like(cs, dtype=np.float64)

    for i, c in enumerate(cs):
        # Get domain update weights
        update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, shape_param)

        T = np.zeros_like(x, dtype=np.float64)

        for t in range(1, max_steps+1):
            T_new = rect_utils.step(T.copy(), update_weights, grid_dist, c, boundary_conditions, boundary_method=rect_utils.unif_boundary)

            print(np.max(np.abs(T_new-T)))
            if np.max(np.abs(T_new - T)) <= convergence_crit:
                convergence_times[i] = t
                converged_vals[i] = T_new[-11,-1]
                break

            T = T_new

    print(converged_vals)
    print(convergence_times)

    plt.scatter(cs, converged_vals)

    plt.xlabel("Number of steps")
    plt.ylabel("Value")
    plt.title("Conergence value of node at (0.6, 0.2) against shape parameter")
    plt.legend()
    plt.axhline(LIMIT, c='r', linestyle='--')
    plt.show()


def gauss_inf_domain_errs(diffusivity, time_step, x_min, x_max, y_min,
                                   y_max, grid_dist, shape_param, num_steps):
    """
    Computes solution for the infinite domain problem with Gaussian initial condition
    and returns the vector of average absolute errors for each timestep

    N = 5
    """
    # Analytical Gaussian solution for comparison
    true_sol = lambda t: np.exp(-(x**2+y**2)/(4*diffusivity*t+1))/(4*diffusivity*t+1)

    # create the grid
    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)

    # Initial condition
    T = np.exp(-(x**2 + y**2))

    # Get domain update weights
    update_weights = general_utils.domain_update_weights(np.array([0, grid_dist, -grid_dist, grid_dist*1j, -grid_dist*1j]), time_step, diffusivity, shape_param)

    errs = np.zeros(num_steps+1, dtype=np.float64)
    t1 = time.time()

    # Iterate
    for i in range(1,num_steps+1):
        # Shrink grid since we are not computing boundary node values
        x, y = x[1:-1,1:-1], y[1:-1,1:-1]
        T = rect_utils.step_domain(T, update_weights)
        print(f"t={i*time_step} max abs error: {np.max(np.abs(T - true_sol(i*time_step)))}")

        abs_diff = np.abs(T-true_sol(i*time_step))
        errs[i] = np.mean(abs_diff)

    t2 = time.time()
    print(f"Total computation time: {t2-t1:.3f} seconds")

    return errs


def gaussian_inf_domain_plot(time_step=0.001, x_min=-6, x_max=6, y_min=-6,
                             y_max=6, grid_dist=0.01, shape_param=4, num_steps=20):
    """
    Log-log plot of average absolute errors for the infinite problem with
    Gaussian initial condition, on a rectangular uniform grid
    """
    params = locals()
    # physical parameters
    rho = 40
    param_c = 46
    k = 52

    # paper Values
    # rho = 7850
    # param_c = 460
    # k = 52
    params["diffusivity"] = k/(rho*param_c)

    errs = gauss_inf_domain_errs(**params)

    plt.semilogy(errs)
    plt.title(f"Avg absolute error over time\nN=5, grid_dist={params['grid_dist']},\
     time_step={params['time_step']}, k={params['diffusivity']}")
    plt.xlabel("Time steps")
    plt.ylabel("Avg abs error (log10 scale)")

    plt.show()


def plot_sarler_second_analytical(t, trunc=50, diff=0.2):
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    grid_dist = 0.025

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1] # by default meshgrid returns a y that increases going down the matrix
                # ie. the opposite of the intuitive picture - need to reverse this
                # decide whether to do this here or just remember how it works and change the
                # boundary conditions implementation

    fig = plt.figure(figsize=(12,6))
    ax = fig.gca(projection='3d')
    sol = analyticals.sarler_second(x, y, t, diff, trunc=trunc)
    surface = ax.plot_surface(x, y, sol)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(surface)
    ax.set_title(f'Truncated analytical solution at t={t}')
    plt.show()
#
