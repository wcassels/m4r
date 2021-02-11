"""
Functions to generate plots! (Will fill up when I copy across old code)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time

import solve_grid
import rect_utils
import general_utils
import analyticals


def sarler_first_test(time_step=1.0e-4, num_steps=50, plot_every=20, shape_param=4):
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
        solve_grid.grid_step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=solve_grid.unif_boundary)

        if (t % plot_every) == 0:
            print(f"NAFEMs convergence val: {T[-11,-1]}")
            fig = plt.figure(figsize=(12,6))
            ax = fig.gca(projection='3d')
            surface = ax.plot_surface(x, y, T)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'RBF solution')
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
        solve_grid.grid_step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=solve_grid.unif_boundary)
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


def second_test_avg_errs(time_step=1.0e-4, num_steps=50, trunc=50, shape_param=4):
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

    errs = np.zeros(num_steps+1, dtype=np.float64)

    for t in range(1, num_steps+1):
        solve_grid.grid_step(T, update_weights, grid_dist, shape_param, boundary_conditions, boundary_method=solve_grid.unif_boundary)

        trunc_sol = analyticals.sarler_second(x, y, t*time_step, diffusivity, trunc=trunc)

        # Correct corners - this has no effect on RBF solution marching
        T[0,0], T[0,-1], T[-1,0], T[-1,-1] = trunc_sol[0,0], trunc_sol[0,-1], trunc_sol[-1,0], trunc_sol[-1,-1]

        errs[t] = np.mean(np.abs(T-trunc_sol))

    plt.plot(range(num_steps+1), errs)
    plt.xlabel("Time steps")
    plt.ylabel("Error")
    plt.title(f"Average solution error (Δt={time_step})")
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
            T_new = solve_grid.grid_step(T.copy(), update_weights, grid_dist, c, boundary_conditions, boundary_method=solve_grid.unif_boundary)

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
        T = solve_grid.step_domain(T, update_weights)
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
