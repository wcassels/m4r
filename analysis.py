"""
Functions to generate plot! (Will fill up when I copy across old code)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import solve_grid
import rect_helpers

def plot_indices_excl_corners(x, y):
    """
    Used for plotting when corner values have not been computed.

    This didn't work :D
    """
    x_list, y_list = np.ravel(np.arange(x)), np.ravel(np.arange(y))

    return np.delete(x_list, [0, x-1, x*(y-1), x*y-1]), np.delete(y_list, [0, x-1, x*(y-1), x*y-1])

def boundary_test_case(time_step=1.0e-5, num_steps=100000, plot_every=1000):
    # second test
    x_min, x_max = 0, 1.0
    y_min, y_max = 0, 1.0

    rho, param_c, k = 5, 1, 1
    # rho, param_c, k = 7850, 460, 52
    diffusivity = k / (rho * param_c)

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
    # T = np.exp(-(x**2+y**2))+100

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surface = ax.plot_surface(x_plot, y_plot, T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # fig.colorbar(surface)
    # plt.title(f'sol')
    # plt.show()

    # Get the collocation matrix
    Phi = rect_helpers.get_Phi(5, grid_dist, shape_param)

    # Get simplified update weights - only works for uniform configuration!
    update_weights = rect_helpers.get_update_weights(Phi, diffusivity, shape_param)

    for t in range(1, num_steps+1):
        solve_grid.grid_step(T, update_weights, grid_dist, shape_param, boundary_conditions)
        # Corner values are not computed. They also have no influence on future calculations
        # so until I figure out how to do 3D plots without them, I am just setting them to
        # be nice values :)
        T[0,0], T[0,-1], T[-1,0], T[-1,-1] = T[1,1], T[1,-2], T[-2,1], T[-2,-2]
        # print(T[-3:,-3:])
        # print(np.unravel_index(np.argmax(np.abs(T), axis=None), T.shape))

        if (t % plot_every) == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surface = ax.plot_surface(x, y, T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('x')
            plt.ylabel('y')
            fig.colorbar(surface)
            plt.title(f'sol')
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
    Phi = rect_helpers.get_Phi(5, grid_dist, shape_param)

    # Initial condition
    T = np.exp(-(x**2 + y**2))

    # Get simplified update weights - only works for uniform configuration!
    update_weights = rect_helpers.get_update_weights(Phi, diffusivity, shape_param)

    errs = np.zeros(num_steps+1, dtype=np.float64)
    t1 = time.time()

    # Iterate
    for i in range(1,num_steps+1):
        # Shrink grid since we are not computing boundary node values
        x, y = x[1:-1,1:-1], y[1:-1,1:-1]
        T = step_domain(T, update_weights)
        print(f"t={i*time_step} error: {np.linalg.norm(T - true_sol(i*time_step))}")

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


def sarler_second_case_analytical(x, y, diff, trunc=50):
    """
    Returns the truncated analytical solution described in the second test of
    the Sarler paper.
    """


def sarler_second_case_nth(x, y, diff, n):
    """
    Returns the nth term in the sum for the analytical solution described in
    the second test of the Sarler paper.
    """
    
