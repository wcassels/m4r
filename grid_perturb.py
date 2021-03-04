import numpy as np
import matplotlib.pyplot as plt

import general_utils
import normal_derivs
import analyticals

time_step = 0.0001
diffusivity = 1
shape_param = 4

x_line = np.arange(0, 1.025, 0.025)
y_line = np.arange(0, 1.025, 0.025)

# unperturbed
x_base, y_base = np.meshgrid(x_line, y_line)

# set positions
positions_base = (x_base + y_base*1j).ravel()

# set initial condition
T = np.ones_like(positions_base, dtype=np.float64)

# define boundaries
labels = np.full(positions_base.size, None)
boundary_vals = np.full(positions_base.size, None)
deriv_lambdas = np.full(positions_base.size, None)

# Second Sarler test case
# North Dirichlet(0) boundary
labels[positions_base.imag == 1] = "D"
boundary_vals[positions_base.imag == 1] = 0
deriv_lambdas[positions_base.imag == 1] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="-")

# South Neumann(0) boundary
labels[positions_base.imag == 0] = "N"
boundary_vals[positions_base.imag == 0] = 0
deriv_lambdas[positions_base.imag == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="y", direction="+")

# West Neumann(0) boundary
labels[positions_base.real == 0] = "N"
boundary_vals[positions_base.real == 0] = 0
deriv_lambdas[positions_base.real == 0] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="+")

# East Dirichlet(0) boundary
labels[positions_base.real == 1] = "D"
boundary_vals[positions_base.real == 1] = 0
deriv_lambdas[positions_base.real == 1] = lambda centre, positions: normal_derivs.cartesian(centre, positions, shape_param, axis="x", direction="-")



num_steps = 1000
check_every = 50
# avg_errs = np.zeros(num_steps // check_every)

for sd in [0, 0.001, 0.0015, 0.002]:
    # set initial condition
    T = np.ones_like(positions_base, dtype=np.float64)

    positions = positions_base.copy()
    print(f"SD: {sd}")
    avg_errs = []

    # perturb
    for i in range(positions.size):
        rands = np.random.normal(0, sd, size=2)
        if labels[i] is None:
            positions[i] += rands[0] + rands[1]*1j

    x, y = positions.reshape(x_base.size).real, positions.reshape(x_base.size).imag

    # get neighbourhoods
    neighbourhood_idx, update_info = general_utils.general_setup(positions, labels, time_step, diffusivity, shape_param)
    for t in range(1, num_steps+1):
        general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        print(t)
        if (t % check_every) == 0:
            avg_errs.append(np.mean(np.abs(T.reshape(x.shape)-analyticals.sarler_second(x, y, t*time_step, diffusivity))))

    plt.plot(np.arange(1, 1+len(avg_errs)) * check_every, avg_errs, label=f"SD={sd}")
# plt.plot(np.arange(1, 1+len(avg_errs))*check_every, avg_errs)
plt.xlabel("Num steps")
plt.ylabel("Error")
plt.title("Average solution error over time for the Second test\nVarying the SD of random node position perturbations")
plt.grid()
plt.legend()
plt.show()
