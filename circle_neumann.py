import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fsolve
from scipy.special import jv

import general_utils
import normal_derivs
import node_utils

# parameters that work:
# time_step 0.0005
# diff .1
# shape_param 16
# num_boundary_nodes 200
# num_domain_nodes 1001
# domain conditions: abs(p) < 1
# mu 2
# guess 4
# reg 4



time_step = 0.0005
diffusivity = .1
shape_param = 160

num_steps = 500

num_boundary_nodes = 200
num_domain_nodes = 1001 # 1501 almost perfect but get unstable spike in just one of the corners

boundary_nodes, labels, boundary_vals, deriv_lambdas = node_utils.make_circle(0.0, 1.0, num_boundary_nodes, "N", 0, lambda centre, pos: normal_derivs.radial(centre, pos, shape_param, direction="inwards"))
# domain_conditions = [lambda p: np.abs(p) < 0.98]
domain_conditions = [lambda p: np.abs(p) < 1]

plt.scatter(boundary_nodes.real, boundary_nodes.imag, s=3)
plt.show()

domain_nodes, cut_outs = node_utils.fill_domain(boundary_nodes, domain_conditions, num_domain_nodes)


# np.save(f"node_positions/disk/{num_boundary_nodes}boundary{num_domain_nodes}domain", nodes)
nodes = np.concatenate((boundary_nodes, domain_nodes))
labels = np.concatenate((np.full(num_boundary_nodes, "N"), np.full(num_domain_nodes, None)))  # due to inner disk
boundary_vals = np.concatenate((np.full(num_boundary_nodes, 0), np.full(num_domain_nodes, None)))
deriv_lambdas = np.concatenate((np.full(num_boundary_nodes, lambda centre, pos: normal_derivs.radial(centre, pos, shape_param)), np.full(num_domain_nodes, None)))

###

## UNIFORM POLAR NODES INCLUDING ON THE BOUNARY
# num_r, num_theta = 9, 25 # 11, 31 is stable but still doesnt look like 0 derivative
# r_pot, theta_pot = np.linspace(0,1,num_r+1)[1:], np.linspace(0, 2*np.pi, num_theta+1)[:-1]
# r, t = np.meshgrid(r_pot, theta_pot)
# r, t = r.ravel(), t.ravel()
# num_domain_nodes = num_theta * (num_r - 1) + 1
# num_boundary_nodes = num_theta
# # nodes = np.concatenate((nodes, r*np.cos(t)+1j*r*np.sin(t)))
# nodes = r*np.cos(t) + 1j* r*np.sin(t)
# r = np.append(r, 0)
# nodes = np.append(nodes, 0+0j)
# plt.scatter(nodes.real, nodes.imag)
# plt.show()
# num_domain_nodes = nodes.size

# labels = np.concatenate((np.full(num_boundary_nodes, "N"), np.full(num_domain_nodes, None)))
# labels = np.full(num_domain_nodes + num_boundary_nodes, None)
# boundary_vals = np.full(num_domain_nodes + num_boundary_nodes, None)
# deriv_lambdas = np.full(num_domain_nodes + num_boundary_nodes, None)
# labels[r==1] = "N"
# boundary_vals[r==1] = 0
# deriv_lambdas[r==1] = lambda centre, pos: normal_derivs.radial(centre, pos, shape_param)

###

# analytial stuff
μ = 2 # from 2
guess = 4

input("i am a banan2a")
# initial guess here has to be fairly good for solve to converge
λ = fsolve(lambda t: jv(μ-1, t) - jv(μ+1, t), guess, xtol=1e-12)[0]
input(f"λ={λ}")

sol = lambda r, theta, t: np.exp(-λ**2 * diffusivity * t) * np.cos(μ * theta) * jv(μ, λ * r)
fig, ax = plt.subplots(1, 2)
for reg in [0,2]:
    print(f"reg={reg}")
    neighbourhood_idx1, update_info1 = general_utils.general_setup(nodes, labels, time_step, diffusivity, shape_param, reg=reg)#, dtype=np.float128)
    # neighbourhood_idx, update_info = general_utils.alternative_setup(nodes, labels, deriv_lambdas, time_step, diffusivity, shape_param, reg=reg)
    neighbourhood_idx2, update_info2 = general_utils.alt_setup_two(nodes, labels, deriv_lambdas, time_step, diffusivity, shape_param, reg=reg)
    # T = np.zeros_like(nodes, dtype=np.float128)
    # T[:] = np.exp(-(np.abs(nodes))**2)
    rs = np.abs(nodes)
    thetas = np.arccos(nodes.real / rs)
    thetas[nodes.imag < 0] *= -1
#
    T = sol(rs, thetas, 0)
    # T = np.exp(-np.abs(nodes)**2)
    T_mod = T.copy()
    T_implicit = T.copy()
    # print(T.dtype)
    #
    # cmin, cmax = np.min(T), np.max(T)
    # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # surf = ax.plot_trisurf(nodes.real, nodes.imag, T)#, vmin=cmin, vmax=cmax, cmap=cm.jet)
    #
    # # ax.set_zlim(cmin, cmax)
    # # ax2.set_zlim(cmin, cmax)
    # # plt.colorbar(surf)
    # # p.set_clim(cmin, cmax)
    # ax.set_title(f"Initial condition")
    #
    # plt.show()

    # plot_cond = lambda t: t*time_step > 0.0
    # plot_cond = lambda t: (t%10) == 0
    plot_cond = lambda t: True
    errs = np.zeros((1+num_steps, 2), dtype=np.float64)



    for t in range(1, num_steps+1):
        general_utils.general_step(T, update_info1, neighbourhood_idx1, labels, boundary_vals, deriv_lambdas, shape_param, t*time_step)
        # general_utils.alternative_step(T, update_info, neighbourhood_idx, labels, boundary_vals)
        general_utils.generalised_everywhere_step(T_mod, update_info2, neighbourhood_idx2, labels, boundary_vals)
        # general_utils.modified_implicit_step(T_implicit, update_info2, neighbourhood_idx2, labels, boundary_vals)
        # print(T.dtype)
        errs[t,0] = np.mean(np.abs(T-sol(rs, thetas, t*time_step)))
        errs[t,1] = np.mean(np.abs(T_mod-sol(rs,thetas,t*time_step)))
        # errs[t,2] = np.mean(np.abs(T_implicit-sol(rs,thetas,t*time_step)))
        print(t)
        # if plot_cond(t):
            # SINGLE PLOT
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # surf = ax.plot_trisurf(nodes.real, nodes.imag, T)#, vmin=cmin, vmax=cmax)
            # # plt.colorbar()
            # ax.set_title(f"RBF Solution after {t*time_step:.3f} seconds using modified implicit scheme")
            # plt.show()

            # COMPARISON PLOT
            # cmin, cmax = np.min(sol(rs, thetas, t*time_step)), np.max(sol(rs, thetas, t*time_step))
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(121, projection='3d')
            #
            # surf = ax.plot_trisurf(nodes.real, nodes.imag, T)#, vmin=cmin, vmax=cmax)
            #
            # ax2 = fig.add_subplot(122, projection='3d')
            # # surf2 = ax2.plot_trisurf(nodes.real, nodes.imag, sol(rs, thetas, t*time_step), vmin=cmin, vmax=cmax)
            # surf2 = ax2.plot_trisurf(nodes.real, nodes.imag, T_mod)
            #
            # # ax.set_zlim3d(cmin, cmax)
            # # ax2.set_zlim3d(cmin, cmax)
            # plt.colorbar(surf)
            # # p.set_clim(cmin, cmax)
            # ax.set_title(f"Original scheme")
            # # ax2.set_title(f"Implicit modified scheme\nGMRES tol={1e-12}")
            # ax2.set_title("Modified scheme")
            # plt.suptitle(f"Solution after {t*time_step:.3f} seconds")
            #
            # plt.show()

    ax[0].semilogy(np.arange(1+num_steps), errs[:,0], label=f"λ={reg}")
    ax[1].semilogy(np.arange(1+num_steps), errs[:,1], label=f"λ={reg}")
    # ax[2].semilogy(np.arange(1+num_steps), errs[:,2], label=f"λ={reg}")

ax[0].legend()
ax[1].legend()
# ax[2].legend()
ax[0].grid()
ax[1].grid()
# ax[2].grid()
ax[0].set_xlabel("Num steps")
ax[1].set_xlabel("Num steps")
# ax[2].set_xlabel("Num steps")
ax[0].set_ylabel("Error")
ax[1].set_ylabel("Error")
# ax[2].set_ylabel("Error")
ax[0].set_title("Original scheme")
ax[1].set_title("My modified scheme")
# ax[2].set_title("My modified scheme (implicit implementation)")

plt.suptitle("Comparing average solution error over time between different schemes")
# plt.legend()
# plt.grid()
# plt.title(f"RBF Neumann disk errors for eigenfunction initial condition\ndiff={diffusivity}, c={shape_param}, Δt={time_step}")
plt.show()
