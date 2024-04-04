import numpy as np
from problem.navier_stokes import convection, diffusion
from tqdm import tqdm

BAR_FMT_STR = '{desc:<20.20s}: {percentage:3.0f}% |{bar}| {n_fmt:>5.5s}/{total_fmt:<5.5s} [{elapsed}]'


# Galerkin system coefficients
# Computation of Galerkin system parameters
def find_galerkin_coefficients(mesh, UAvg, PODModes, scalar_product, num_pod_modes=4):

    Qavg = convection(mesh, UAvg, UAvg)
    Lavg = diffusion(mesh, UAvg)

    # initialize arrays for parameters
    b1 = np.empty(num_pod_modes)
    b2 = np.empty(num_pod_modes)
    L1 = np.empty((num_pod_modes, num_pod_modes))
    L2 = np.empty((num_pod_modes, num_pod_modes))
    Q = [np.empty((
        num_pod_modes,
        num_pod_modes,
    )) for x in range(num_pod_modes)]

    # compute L and Q operators for projection
    Q_tmp1 = np.empty((num_pod_modes, 2 * mesh.number_nodes))
    Q_tmp2 = np.empty((num_pod_modes, 2 * mesh.number_nodes))
    Q_tmp3 = np.empty((num_pod_modes, num_pod_modes, 2 * mesh.number_nodes))
    L_tmp = np.empty((num_pod_modes, 2 * mesh.number_nodes))

    for i in tqdm(range(num_pod_modes), ncols=72, desc='Compute L/Q', bar_format=BAR_FMT_STR):
        Q_tmp1[i] = convection(mesh, UAvg, PODModes[:, i])
        Q_tmp2[i] = convection(mesh, PODModes[:, i], UAvg)
        L_tmp[i] = diffusion(mesh, PODModes[:, i])
        for j in range(num_pod_modes):
            Q_tmp3[i, j] = convection(mesh, PODModes[:, i], PODModes[:, j])

    # compute ODE coefficients
    for k in tqdm(range(num_pod_modes), ncols=72, desc='Compute ODE Coeff', bar_format=BAR_FMT_STR):
        b1[k] = scalar_product(Lavg, PODModes[:, k])
        b2[k] = scalar_product(Qavg, PODModes[:, k])
        for i in range(num_pod_modes):
            L1[k, i] = scalar_product(L_tmp[i], PODModes[:, k])
            L2[k, i] = scalar_product(np.add(Q_tmp1[i], Q_tmp2[i]), PODModes[:, k])
            for j in range(num_pod_modes):
                Q[k][i, j] = scalar_product(Q_tmp3[i, j], PODModes[:, k])

    return b1, b2, L1, L2, Q
