import matplotlib.pyplot as plt
import numpy as np
import utils.readers as resread
from pod import find_pod_modes, find_time_coeffs
from projection import find_galerkin_coefficients
from snapshots import Snapshots
from tqdm import tqdm
from utils.plotting import plot_cylinder_data, plot_square_data

BAR_FMT_STR = '{desc:<20.20s}: {percentage:3.0f}% |{bar}| {n_fmt:>5.5s}/{total_fmt:<5.5s} [{elapsed}]'

plotting = True
dimension = 2
num_pod_modes = 8

mesh, snapshots = resread.load_results('./results/1200result', 5)
inner_weights = np.ones(mesh.number_nodes * dimension) * mesh.volume_weights


# weighted scalar product
def scalar_product(var1, var2, weights=inner_weights):
    return np.sum(var1 * var2 * weights)


# calculating mean flow field and fluctuating velocity
def find_avg_fluc(snapshots):
    #snapshot.U = stack of u and v velocity
    Us = np.array([snap.U() for snap in snapshots]).reshape((snapshots.num_snaps, -1))
    # mean flow field
    UAvg = np.mean(Us, 0)
    # centered flow field
    UFluc = Us - np.repeat(np.expand_dims(UAvg, 0), snapshots.num_snaps, axis=0)
    UFluc = UFluc.transpose()
    return UAvg, UFluc


def galerkin_system(t, a, nu=0.01):
    global b1, b2, L1, L2, Q
    a_dot = np.empty_like(a, dtype=np.float128)
    for k in range(a_dot.shape[0]):
        a_dot[k] = nu * b1[k] + b2[k] + np.inner(
            (nu * L1[k, :] + L2[k, :]), a) + np.matmul(np.matmul(np.expand_dims(a, 1).T, Q[k]), np.expand_dims(a, 1))
    return a_dot


# defined ode solver
def RK45(f, interval, a0, dt):
    # get initial and end time
    t = interval[0]
    tmax = interval[1]
    # compute number of steps
    Nt = np.abs(int((tmax - t) / dt)) + 1
    # initialize time coefficients projection matrix
    a = np.zeros((a0.shape[0], Nt))
    # initial conditions
    a[:, 0] = a0
    for i in tqdm(range(Nt - 1), desc='Integration', ncols=72, bar_format=BAR_FMT_STR):
        k1 = dt * f(t, a[:, i])
        k2 = dt * f(t + dt / 2, a[:, i] + k1 / 2)
        k3 = dt * f(t + dt / 2, a[:, i] + k2 / 2)
        k4 = dt * f(t + dt, a[:, i] + k3)
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        a[:, i + 1] = a[:, i] + k
        t = t + dt
    return a


UAvg, UFluc = find_avg_fluc(snapshots)
[PODModes, S] = find_pod_modes(UFluc, scalar_product, num_pod_modes=num_pod_modes)
time_coeffs = find_time_coeffs(UFluc, PODModes, scalar_product)
b1, b2, L1, L2, Q = find_galerkin_coefficients(mesh, UAvg, PODModes, scalar_product, num_pod_modes=num_pod_modes)

a0 = time_coeffs[:num_pod_modes, 0]
time_interval = (0, (snapshots.num_snaps - 1) * 0.25)
dt = 0.25
time_coeffs_projection = RK45(galerkin_system, time_interval, a0, dt)

if plotting:
    # Plotting POD Modes
    xcoord, ycoord = mesh.node_coords()

    for i in range(num_pod_modes):
        # POD modes for velocity u
        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plot_square_data(xcoord, ycoord, PODModes[:mesh.number_nodes, i], resolution=1000)
        # plt.savefig(f'pngs/pod_mode_{i}.png', dpi=300)
        plt.show()

    # Plotting reference time coefficients vs galerkin time coefficients
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    # loop four dofs
    for i in range(0, 1):
        ax.plot(np.linspace(0, dt * len(time_coeffs_projection[i, :]), len(time_coeffs_projection[i, :]))[:],
                time_coeffs_projection[i, :],
                label="Galerkin Projection Time Coefficient " + str(i + 1))
    for i in range(0, 1):
        ax.plot(np.linspace(0, dt * len(time_coeffs[i, :]), len(time_coeffs[i, :])),
                time_coeffs[i, :],
                label="Time Coefficient " + str(i + 1),
                linestyle="dashed")

    ax.legend(title="Projection", loc=1)
    ax.set_xlabel("t", fontsize=16)
    ax.set_ylabel("a", fontsize=16, labelpad=-2)
    # plt.savefig(f'pngs/time_coeffs.png', dpi=300)
    plt.show()

    # reconstruction plotting
    rec = np.zeros_like(UFluc)
    for j in range(4):
        rec += np.outer(PODModes[:, j], time_coeffs_projection[j, :])
    rec += np.repeat(np.expand_dims(UAvg, 1), snapshots.num_snaps, 1)

    for i in range(snapshots.num_snaps):
        if i % 50 == 0:
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            plot_square_data(xcoord, ycoord, rec[:mesh.number_nodes, i], resolution=200, ax=ax, cbar=False)
            # plt.savefig(f'pngs/reconstruction_{i}.png', dpi=300)
            plt.show()
