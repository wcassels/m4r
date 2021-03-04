import numpy as np
import matplotlib.pyplot as plt

import general_utils

# boundaries
inner_lambda = lambda theta: 0.3 + 0.1 * np.sin(theta) + 0.15 * np.sin(5*theta)
outer_lambda = lambda theta: 1 + 0.2 * np.cos(theta) + 0.15 * np.sin(4*theta)

num_inner_domain_nodes = 65

theta = np.linspace(0, 2*np.pi-1e-1, num_inner_domain_nodes)

r_inner = inner_lambda(theta)
r_outer = outer_lambda(theta)

inner_cart = r_inner * np.cos(theta) + 1j* r_inner*np.sin(theta)
outer_cart = r_outer * np.cos(theta) + 1j* r_outer*np.sin(theta)

nodes = np.concatenate((inner_cart, outer_cart))
boundaries = nodes.copy()

thetas = np.linspace(0, 2*np.pi-1e-2, 100)
rs = np.linspace(0, 1.5, 50)

r, theta = np.meshgrid(rs, thetas)
# polars stored in form (r, theta)
polars = (r + theta * 1j).ravel()

polars = polars[polars.real > inner_lambda(polars.imag)]
polars = polars[polars.real < outer_lambda(polars.imag)]

# plt.scatter(polars.real * np.cos(polars.imag), polars.real * np.sin(polars.imag), s=1.5)
# plt.scatter(nodes.real, nodes.imag, s=1.5)
# plt.grid()
# plt.show()

# num_points = 200
cartesians = polars.real * np.cos(polars.imag) + 1j * polars.real * np.sin(polars.imag)
cartesians_base = cartesians.copy()
nodes_base = nodes.copy()

# analytical
sol = lambda x, y, t: np.exp(-5*t*np.pi**2) * np.sin(np.pi*x) * np.sin(2*np.pi*y)

time_step = 0.001
diffusivity = 1
# shape_param = 4
num_points = 100

cartesians = cartesians_base.copy()
nodes = nodes_base.copy()

for i in range(num_points):
    print(i)
    ds = np.zeros_like(cartesians, dtype=np.float64)

    # vectorize this
    for j in range(cartesians.size):
        ds[j] = np.min(np.abs(nodes - cartesians[j]))

    k = np.argmax(ds)
    nodes = np.append(nodes, cartesians[k])
    cartesians = np.delete(cartesians, k)

# plt.scatter(nodes.real, nodes.imag, s=3, c='red')
# plt.scatter(boundaries.real, boundaries.imag, s=3, c='green')
# plt.title(f"{i+1} domain nodes added")
# plt.grid()
# plt.show()

positions = nodes.copy()

labels = np.concatenate((np.full(num_inner_domain_nodes * 2, "D-f"), np.full(num_points, None)))
boundary_vals = np.concatenate((np.full(num_inner_domain_nodes * 2, sol), np.full(num_points, None)))
deriv_lambdas = np.full(num_inner_domain_nodes * 2 + num_points, None)

shape_param = 12
# for shape_param in [0.5, 1, 4, 12, 48]:
for enforce_sum in [False, True]:
    print(shape_param)
    neighbourhood_idx, update_info = general_utils.general_setup(positions, labels, time_step, diffusivity, shape_param, enforce_sum=enforce_sum)

    # initial condition`
    T = sol(positions.real, positions.imag, 0)

    cmin, cmax = np.min(T), np.max(T)

    num_steps = 200

    errs = np.zeros(num_steps+1, dtype=np.float64)

    for t in range(1, num_steps+1):
        general_utils.general_step(T, update_info, neighbourhood_idx, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        true_sol = sol(positions.real, positions.imag, t*time_step)

        if (t % 10) == 0 and num_points > 100:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            rbf = ax[0].scatter(positions.real, positions.imag, c=T, s=3)
            fig.colorbar(rbf, ax=ax[0])
            rbf.set_clim(cmin, cmax)
            ax[0].set_title(f"RBF Solution after {t * time_step:.3f} seconds\n{num_points} domain nodes, Δt={time_step}, diff={diffusivity}")

            true = ax[1].scatter(positions.real, positions.imag, c=true_sol, s=3)
            fig.colorbar(true, ax=ax[1])
            true.set_clim(cmin, cmax)
            ax[1].set_title(f"Analytical solution")

            plt.show()


        errs[t] = np.mean(np.abs(true_sol-T))


    plt.plot(range(num_steps+1), errs, label="Sum enforced" if enforce_sum else "Standard")

plt.xlabel("Num steps")
plt.ylabel("Error")
plt.title("Avg abs error over time")
plt.legend()
plt.grid()
plt.show()
