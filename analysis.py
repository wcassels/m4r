"""
Functions to generate plots! (Will fill up when I copy across old code)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time

import solve_grid
import unif_utils


def sarler_first_test():
    # write this :)

def sarler_second_test(time_step=1.0e-4, num_steps=50, plot_every=20, trunc=50, diff=None):
    # second test
    x_min, x_max = 0, 1.0
    y_min, y_max = 0, 1.0

    if diff is None:
        rho, param_c, k = 1, 1, 1
        # rho, param_c, k = 7850, 460, 52
        diffusivity = k / (rho * param_c)
    else:
        diffusivity = diff

    shape_param = 16

    # test 1
    # boundary_conditions = {"North": ("Robin", (-750/k, 0)), "East": ("Robin", (-750/k, 0)),
    #                        "South": ("Dirichlet", (100, None)), "West": ("Neumann", (0, None))}

    # test 2
    boundary_conditions = {"North": ("Dirichlet", (0, None)), "East": ("Dirichlet", (0, None)),
                            "South": ("Neumann", (0, None)), "West": ("Neumann", (0, None))}

    grid_dist = 0.025
    # time_step = 0.01 # change
    # num_steps = 100000

    x_line = np.arange(x_min, x_max + grid_dist, grid_dist)
    y_line = np.arange(y_min, y_max + grid_dist, grid_dist)
    x, y = np.meshgrid(x_line, y_line)
    y = y[::-1] # by default meshgrid returns a y that increases going down the matrix
                # ie. the opposite of the intuitive picture - need to reverse this
                # decide whether to do this here or just remember how it works and change the
                # boundary conditions implementation

    # x_idx, y_idx = plot_indices_excl_corners(x_line.size, y_line.size)

    # Initial condition
    T = np.ones_like(x, dtype=np.float64)

    # Get the collocation matrix
    Phi = unif_utils.get_Phi(5, shape_param)

    # Get simplified update weights - only works for uniform configuration!
    update_weights = unif_utils.get_update_weights(Phi, diffusivity, time_step, grid_dist, shape_param)

    for t in range(1, num_steps+1):
        solve_grid.grid_step(T, update_weights, grid_dist, shape_param, boundary_conditions)
        # Corner values are not computed. They also have no influence on future calculations
        # so until I figure out how to do 3D plots without them, I am just setting them to
        # be nice values :)
        # T[0,0], T[0,-1], T[-1,0], T[-1,-1] = T[1,1], T[1,-2], T[-2,1], T[-2,-2]

        if (t % plot_every) == 0:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            surface = ax.plot_surface(x, y, T, linewidth=0, antialiased=False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'RBF solution')

            trunc_sol = sarler_second_case_analytical(x, y, t*time_step, diffusivity, trunc=trunc)
            ax = fig.add_subplot(1, 2, 2, projection="3d")
            surface = ax.plot_surface(x, y, trunc_sol)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Truncated analytical solutiones')
            plt.show()

    return


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

    # Get the collocation matrix
    Phi = unif_utils.get_Phi(5, shape_param)

    # Initial condition
    T = np.exp(-(x**2 + y**2))

    # Get simplified update weights - only works for uniform configuration!
    update_weights = unif_utils.get_update_weights(Phi, diffusivity, time_step, grid_dist, shape_param)

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


def sarler_second_case_analytical(x, y, t, diff, trunc=50, dist=1):
    """
    Returns the truncated analytical solution described in the second test of
    the Sarler paper.
    """
    sol = np.zeros_like(x)
    m, n = x.shape

    # term_idx = np.arange(1, trunc+1)
    term_idx = np.arange(trunc) # Sarler paper has a typo :O

    for i in range(m):
        for j in range(n):
            x_terms = sarler_second_case_nth(x[i,j], t, diff, dist, term_idx)
            y_terms = sarler_second_case_nth(y[i,j], t, diff, dist, term_idx)
            sum_x = 4 * np.sum(x_terms) / np.pi
            sum_y = 4 * np.sum(y_terms) / np.pi

            sol[i,j] = sum_x * sum_y

    return sol


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
    sol = sarler_second_case_analytical(x, y, t, diff, trunc=trunc)
    surface = ax.plot_surface(x, y, sol)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(surface)
    ax.set_title(f'Truncated analytical solution at t={t}')
    plt.show()


def sarler_second_test_nth(eta, t, diff, dist, n):
    """
    Returns the nth term of either sum for the analytical solution described in
    the second test of the Sarler paper.

    Arguments:
    - eta: x or y
    - t: time value
    - diff: diffusivity of the problem
    - dist: eta_max - eta_min
    - n: the term in the sum to compute
    """
    # changed eta-1 to eta?
    return ((-1) ** n / (2*n + 1)) * np.exp(-(diff*(2*n+1)**2*np.pi**2*t) / (4*dist**2)) * np.cos((2*n+1)*np.pi*(eta)/(2*dist))
