import numpy as np


# convection operator
def convection(mesh, var1, var2):
    n = int(len(var1) / 2)
    # local variable vectors
    u1 = var1[:n]
    v1 = var1[n:2 * n]
    u2 = var2[:n]
    v2 = var2[n:2 * n]
    # derivatives
    [du2dx, du2dy] = mesh.finite_differences(u2)
    [dv2dx, dv2dy] = mesh.finite_differences(v2)
    # state based convection
    u_tmp = u1 * du2dx + v1 * du2dy
    v_tmp = u1 * dv2dx + v1 * dv2dy
    return -1 * np.concatenate((u_tmp, v_tmp))


# diffusion operator
def diffusion(mesh, var):
    n = int(len(var) / 2)
    # local variable vectors
    u = var[:n]
    v = var[n:2 * n]
    # derivatives
    [_, _, ulap] = mesh.finite_differences(u, compute_laplacian=True)
    [_, _, vlap] = mesh.finite_differences(v, compute_laplacian=True)
    # state based diffusion
    return np.concatenate((ulap, vlap))