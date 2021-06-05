import numpy as np
import rect_utils
import general_utils
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mpmath

x_min, x_max, dx = -2, 2, 0.1
y_min, y_max, dy = -2, 2, 0.1

num_steps = 100
diffusivity = .1
dt = .01
shape_param = 4

x_range = np.arange(-2-num_steps * dx, 2 + (num_steps + 1) * dx, dx)
y_range = np.arange(-2-num_steps * dy, 2 + (num_steps + 1) * dy, dy)

x, y = np.meshgrid(x_range, y_range)

T = np.exp(-(x**2 + y**2))
true_sol = lambda x_, y_, t_: np.exp(-(x_**2+y_**2)/(4*diffusivity*t_+1))/(4*diffusivity*t_+1)

# plot_cond = lambda t: (t % 10) == 0
# plot_cond = lambda t: t == 100
plot_cond = lambda t: False

# shape_params = [1, 4, 16, 64, 128, 256]
# shape_params = range(1, 302, 5)
#
# # fig = plt.figure()
# # end_errs = np.zeros_like(shape_params, dtype=np.float64)
# conds = np.zeros_like(shape_params, dtype=np.float64)
# w0 = np.zeros_like(shape_params, dtype=np.float64)
# w1 = np.zeros_like(shape_params, dtype=np.float64)
#
# for i, shape_param in enumerate(shape_params):
#     print(i)
#     T = np.exp(-(x**2 + y**2))
#     weights = general_utils.domain_update_weights(np.array([0, 1, -1j, 1j, -1]) * dx, dt, diffusivity, shape_param)
#     w0[i] = weights[0]
#     w1[i] = weights[1]
#     conds[i] = general_utils.phi_cond(np.array([0, 1, -1j, 1j, -1]) * dx, dt, diffusivity, shape_param)
    # errs = np.zeros(num_steps+1, dtype=np.float64)
    # for t in range(1, num_steps+1):
    #     # T = rect_utils.domain_step(T, weights)
    #     rect_utils.domain_step(T, weights)
    #     print(shape_param, t)
    #     # print(T[num_steps:-num_steps,num_steps:-num_steps].shape)
    #     # print(true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt).shape)
    #     # rect_utils.domain_step(T, weights)
    #     # T = T[1:-1,1:-1]
    #     # errs[t] = np.mean(np.abs(T[num_steps:-num_steps,num_steps:-num_steps]-true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt)))
    #     if plot_cond(t):
    #         true = true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt)
    #         fig = plt.figure()
    #         ax = fig.gca(projection='3d')
    #         ax.plot_surface(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], T[num_steps-t:-num_steps+t,num_steps-t:-num_steps+t])
    #         # ax.plot_surface(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], true)
    #         # ax.plot_surface(x, y, T)
    #         # plt.title(f"t={t}")
    #         plt.xlabel("x")
    #         plt.ylabel("y")
    #         ax.set_zlabel("T")
    #         plt.tight_layout()
            # plt.savefig("report_figs/gaussian_example.eps", format="eps")
            # plt.show()

    # end_errs[i] = np.mean(np.abs(T[num_steps:-num_steps,num_steps:-num_steps]-true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt)))
    # plt.semilogy(range(num_steps+1), errs, label=f"c={shape_param}")
#
# print(end_errs)
# np.save("end_errs.npy", end_errs)
# plt.semilogy(shape_params, end_errs)
# plt.xlabel("Shape parameter")
# plt.ylabel("Error")
# plt.grid()
# plt.tight_layout()
# plt.savefig("report_figs/gaussian_shape_params_end_errs.eps", format="eps")

# plt.xlabel("Time steps")
# plt.ylabel("Error")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# # plt.savefig("report_figs/gaussian_shape_params_over_time.eps", format="eps")
# print(conds)
# plt.figure()
# ax = plt.gca()
# # # errs = np.load("end_errs.npy")
# plt.plot(shape_params, w0, label="Central weight")
# # plt.plot(shape_params, w1, label="Outer weight")
# # plt.semilogy(shape_params, errs)
# plt.legend()
# plt.xlabel("Shape parameter")
# plt.ylabel("Weight")
# ticks = ax.get_xticks()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
# print(ticks)
# plt.ylim(-0.4001,-.3999)
# plt.grid()
# plt.tight_layout()
# # plt.show()
# # plt.savefig("report_figs/banana.eps", format="eps")
# # plt.savefig("report_figs/banana")
#
# # plt.plot(np.arange(5), np.arange(5))
# plt.savefig("report_figs/gaussian_weights_cut.pdf", format="pdf")

# dx/dt accuracy plot
# dxs = np.arange(0.2, 1.05, 0.05)
dx = 0.2
# dt = 0.005
dts = np.arange(0.0025, 0.00525, 0.00025)
diffusivity = 1
shape_param = 100
# num_steps = 500
# errs = np.zeros_like(dxs, dtype=np.float64)
errs = np.zeros_like(dts, dtype=np.float64)
# for i, dx in enumerate(dxs):

for i, dt in enumerate(dts):
    weights = general_utils.sarler_domain_weights(np.array([0, dx *1j, -dx, dx, -dx*1j]), dt, diffusivity, shape_param)
    # neighbourhood_idx, weights = general_utils.setup(nodes, np.full(nodes.size, None), np.full(nodes.size, None), np.full(nodes.size, None), dt, diffusivity, shape_param, 5, method="Sarler")


    num_steps = int(np.ceil(0.5/dt))
    x_range = np.arange(-2-num_steps * dx, 2 + (num_steps + 1) * dx, dx)
    x, y = np.meshgrid(x_range, x_range)
    T = np.exp(-(x**2 + y**2))
    nodes = (x+1j*y).ravel()
    print(num_steps)
    # input()

    for t in range(1, num_steps+1):
        # T = general_utils.step(T.ravel(), weights, neighbourhood_idx, np.full(T.size, None), np.full(T.size, None), method="Sarler implicit").reshape(*x.shape)
        T[1:-1,1:-1] += weights[0] * T[1:-1,1:-1] + weights[1] * (T[:-2,1:-1]
                        + T[2:,1:-1] + T[1:-1,:-2] + T[1:-1,2:])
        print(dx, dt, t)

        err = np.max(np.abs(T[num_steps:-num_steps,num_steps:-num_steps]-true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt)))
        print(err)
        # input()
    errs[i] = err


np.save(f"data/infinite/dt_err_ext_max", errs)
plt.semilogy(dts, errs)
plt.grid()
plt.xlabel("$\Delta t$")
plt.ylabel("Error")
plt.savefig("report_figs/gaussian/dt_err_ext_max.pdf", format="pdf")


# # STABILITY PLOT CODE
# dxs = np.arange(0.05, 0.11, 0.01)
# # # dxs = [0.01]
# t_crits = np.zeros_like(dxs, dtype=np.float64)
# diffusivity = 1
# shape_param = 100
# dps = 30
# mpmath.mp.dps = dps
#
#
# for i, dx in enumerate(dxs):
#     x_range = np.arange(-2-num_steps * dx, 2 + (num_steps + 1) * dx, dx)
#     x, y = np.meshgrid(x_range, x_range)
#     print(x.size)
#     # x_mp, y_mp = mpmath.matrix(x), mpmath.matrix(y)
#     print("hi!")
#     nodes = (x+1j*y).ravel()
#     neighbourhood_idx, weights_base = general_utils.setup(nodes, np.full(nodes.size, None), np.full(nodes.size, None), np.full(nodes.size, None), dt, diffusivity, shape_param, 5, method="Sarler")
#     np.save("implicit_idx", neighbourhood_idx)
#     np.save("implici_weights_base", weights_base)
#
#     # for dt in np.arange(3e-3+1e-5,1e-5,-1e-5):
#     for dt in np.arange(3e-3+1e-5, 1e-5, -1e-5):
#         T = np.exp(-(x**2 + y**2)).ravel()
#         # print(T.shape)
#         # T = mpmath.matrix(T)
#         # print("hi2")
#         # weights = general_utils.mpmath_domain_update_weights(np.array([0, 1, -1j, 1j, -1]) * dx, dt, diffusivity, shape_param, dps=dps)
#         # weights = np.float128(weights)
#         weights = weights_base * dt
#         print(weights)
#
#         for t in range(1, num_steps+1):
#             # rect_utils.domain_step(T, weights)
#             T = general_utils.step(T.ravel(), weights, neighbourhood_idx, np.full(T.size, None), np.full(T.size, None), method="Sarler implicit").reshape(*x.shape)
#             # T[1:-1,1:-1] += weights[0] * T[1:-1,1:-1] + weights[1] * (T[:-2,1:-1]
#             #                 + T[2:,1:-1] + T[1:-1,:-2] + T[1:-1,2:])
#             print(dx, dt, t)
#             # print(T[num_steps:-num_steps,num_steps:-num_steps].shape)
#             # print(true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt).shape)
#             # rect_utils.domain_step(T, weights)
#             # T = T[1:-1,1:-1]
#             err = np.max(np.abs(T[num_steps:-num_steps,num_steps:-num_steps]-true_sol(x[num_steps:-num_steps,num_steps:-num_steps], y[num_steps:-num_steps,num_steps:-num_steps], t*dt)))
#
#             if err > 1:
#                 print("unstable!")
#                 break
#         else:
#             print("remained stable!")
#             t_crits[i] = dt
#             np.save(f"stability_{dx}_implicit.npy", dt)
#             break
# #

# np.save("fd_stability_vals.npy", t_crits)
# t_crits_fd = np.load("fd_stability_vals.npy")
# t_crits = np.load("stability_vals.npy")
# t_crits0 = np.load("stability_0.01.npy")
# t_crits = np.load("banana.npy")
# t_crits = np.concatenate((t_crits0, t_crits))
# t_crits = np.insert(t_crits, 0, t_crits0)
# np.save("stability_vals_128.npy", t_crits)
# t_crits = np.load("stability_vals.npy")
# t_crits128 = np.load("stability_vals_128.npy")
# np.save("banana.npy", t_crits)
# print(t_crits)
# plt.plot(dxs, t_crits, label="c=100")
# plt.plot(dxs, t_crits_fd, label="FD")
# plt.plot(dxs, t_crits128, label="128")
# plt.plot(dxs, dxs**2/2)
# plt.grid()
# plt.xlabel("Δx")
# plt.ylabel("Δt")
# plt.tight_layout()
# plt.legend()
# plt.savefig("report_figs/gaussian_stability.pdf", format="pdf")
# plt.show()
