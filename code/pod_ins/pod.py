import numpy as np


def find_pod_modes(snapshots, scalar_product, num_pod_modes=4):

    num_snaps = snapshots.shape[1]
    num_total_nodes = snapshots.shape[0]  # 2 * num_nodes (for u and v)

    # correlation matrix
    C = np.empty((num_snaps, num_snaps))
    for i in range(num_snaps):
        # utilize symmetry of C
        for j in range(i, num_snaps):
            C[i, j] = scalar_product(snapshots[:, i], snapshots[:, j])
            C[j, i] = C[i, j]  # symmetry property

    # eigenvalue problem of correlation matrix
    S, V = np.linalg.eigh(C, UPLO='L')
    # flip due to return structure of bp.linalg.eigh
    S = np.flip(S, 0)  # make S in descending order
    V = np.flip(V, 1)  # make V correspondingly

    # construct spatial POD Modes from snapshots
    PODModes = np.zeros((num_total_nodes, num_pod_modes))
    for i in range(num_pod_modes):
        PODModes[:, i] = 1 / np.sqrt(S[i]) * np.matmul(snapshots, V[:, i])

    # computing eigenvalues from snapshots
    S = np.zeros(num_pod_modes)
    for i in range(num_pod_modes):
        for j in range(num_snaps):
            S[i] += scalar_product(snapshots[:, j], PODModes[:, i])**2

    return [PODModes, S]


def find_time_coeffs(snapshots, podModes, scalar_product, reconstruction_dimension=0):

    # input dimensions
    num_total_nodes = podModes.shape[1]
    num_snaps = snapshots.shape[1]

    # if reconstruction dimension is 0 -> use all possible vectors
    if reconstruction_dimension == 0:
        reconstruction_dimension = num_total_nodes

    # compute mode activation by projection
    time_coeffs = np.zeros((num_total_nodes, num_snaps))
    for t in range(num_snaps):
        for i in range(reconstruction_dimension):
            time_coeffs[i, t] = scalar_product(snapshots[:, t], podModes[:, i])

    return time_coeffs