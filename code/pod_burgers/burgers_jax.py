import matplotlib.pyplot as plt
from jax import vmap, jit
import jax.lax as lax
import jax.numpy as jnp
from jax.config import config as jaxconfig
from functools import partial
import numpy as np
from matplotlib import cm
import jax.experimental.ode as jode

jaxconfig.update("jax_enable_x64", True)

# Snapshot Input Variables

nx = 1025  # Number of grid points in x-direction
Lx = 1.0  # Domain length in x-direction
dx = Lx / (nx - 1)  # Distance between two grid points

nRe = 4  # Number of Reynolds numbers
Re_Init = 100.0  # Initial Reynolds number
Re_Final = 1000.0  # Final Reynols number

T = 1.0  # Final time
nSnap = 101  # Number of snapshot per each Re
nTotalSnap = nSnap * nRe  # Total number of snapshots

# Galerkin Projection Input Variables

nt = 500
nModes = 6  # Number of POD modes
dt = T / nt  # Time step
rgk3_time_domain = jnp.linspace(0, T, nt + 1)


# Calculates the exact solution u to the equation for given x, t, and nu VALUES.
@jit
def u(x, t, nu):
    A = (x / (t + 1.0))
    B = jnp.sqrt((t + 1.0) / jnp.exp(1.0 / (8.0 * nu)))
    C = jnp.exp(x * x / (4.0 * nu * (t + 1.0)))
    u = A / (1.0 + B * C)
    return u


# Calculates the exact solution to the equation for given x, t, and nu DOMAINS.
@jit
def calculate_exact_solution(x_domain, t_domain, nu_domain):

    @jit
    def nu_loop(x, t, nu_domain):
        return vmap(u, in_axes=(None, None, 0))(x, t, nu_domain)

    @jit
    def t_loop(x, t_domain, nu_domain):
        return vmap(nu_loop, in_axes=(None, 0, None))(x, t_domain, nu_domain)

    @jit
    def x_loop(x_domain, t_domain, nu_domain):
        return vmap(t_loop, in_axes=(0, None, None))(x_domain, t_domain, nu_domain)

    exact_solution = x_loop(x_domain, t_domain, nu_domain)
    return exact_solution


# Creates the snapshot to the equation for given x, t, and nu DOMAINS.
@jit
def create_snapshot(x_domain, t_domain, nu_domain):

    @jit
    def t_loop(x, t_domain, nu):
        return vmap(u, in_axes=(None, 0, None))(x, t_domain, nu)

    @jit
    def nu_loop(x, t_domain, nu_domain):
        return vmap(t_loop, in_axes=(None, None, 0))(x, t_domain, nu_domain)

    @jit
    def x_loop(x_domain, t_domain, nu_domain):
        return vmap(nu_loop, in_axes=(0, None, None))(x_domain, t_domain, nu_domain)

    snapshot = x_loop(x_domain, t_domain, nu_domain)
    return snapshot


# Calculates time correlation matrix
@jit
def calculate_correlation_matrix(fluc_i, fluc_j, x_domain):

    @jit
    def integrate(fluc_i, fluc_j, x_domain):
        return jnp.trapz(fluc_i * fluc_j, x_domain)

    @jit
    def j_loop(fluc_i, fluc_j, x_domain):
        return vmap(integrate, in_axes=(None, 1, None))(fluc_i, fluc_j, x_domain)

    @jit
    def i_loop(fluc_i, fluc_j, x_domain):
        return vmap(j_loop, in_axes=(1, None, None))(fluc_i, fluc_j, x_domain)

    correlation = i_loop(fluc_i, fluc_j, x_domain)
    return correlation


x_domain = jnp.linspace(0, Lx, nx)
t_domain = jnp.linspace(0, T, nSnap)
re_domain = jnp.linspace(Re_Init, Re_Final, nRe)
nu_domain = 1.0 / re_domain

exact_solution = calculate_exact_solution(x_domain, t_domain, nu_domain)
snapshot = create_snapshot(x_domain, t_domain, nu_domain)
snapshot = snapshot.reshape(nx, nTotalSnap)

# Calculate mean velocity
mean = snapshot.mean(axis=1)

# Calculate fluctuating velocity
subtract_mean = jit(lambda snapshot, mean: snapshot - mean)
fluc = vmap(subtract_mean, in_axes=(0, 0))(snapshot, mean)

# Calculate correlation matrix
correlation = calculate_correlation_matrix(fluc, fluc, x_domain)

# Find eigenvalues and eigenvectors
eig_vals, eig_vecs = jnp.linalg.eigh(correlation)
eig_vals = jnp.real(eig_vals)
eig_vecs = jnp.real(eig_vecs)

idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]
eig_vals_reduced = eig_vals[:nModes]
eig_vecs_reduced = eig_vecs[:, :nModes]

# Find POD modes by taking the dot product of fluctuating velocity and eigenvectors
uPOD = jnp.dot(fluc, eig_vecs_reduced, precision=lax.Precision.HIGHEST)
# Normalize eigenvectors to unity
uPOD = uPOD / jnp.sqrt(jnp.abs(eig_vals_reduced.transpose()))

# Galerkin Projection

re_test = 750.0
nu_test = 1.0 / re_test
nu_test_domain = jnp.array([nu_test])

# Exact solution for the test case
snapshot_test = create_snapshot(x_domain, rgk3_time_domain, nu_test_domain)
# Exact solution for fluctuating part of test case
fluc_test = vmap(subtract_mean, in_axes=(0, 0))(snapshot_test, mean)
fluc_test = fluc_test[:, 0, :]


# Calculate the true time coefficients obtained from the projecting the snapshot matrix into POD modes
@jit
def calculate_time_coeffs(uPOD, fluc_test):

    @jit
    def calculate(uPOD_k, fluc_test_i):
        return jnp.trapz(uPOD_k * fluc_test_i, x_domain)

    @jit
    def t_loop(uPOD_k, fluc_test):
        return vmap(calculate, in_axes=(None, 1))(uPOD_k, fluc_test)

    @jit
    def modes_loop(uPOD, fluc_test):
        return vmap(t_loop, in_axes=(1, None))(uPOD, fluc_test)

    time_coeffs = modes_loop(uPOD, fluc_test)
    return time_coeffs


time_coeffs = calculate_time_coeffs(uPOD, fluc_test)

meandx = jnp.gradient(mean, dx)
meanddx = jnp.gradient(meandx, dx)

uPODdx = vmap(jnp.gradient, in_axes=(1, None), out_axes=1)(uPOD, dx)
uPODddx = vmap(jnp.gradient, in_axes=(1, None), out_axes=1)(uPODdx, dx)


@jit
def calculate_constant_term(uPOD):

    @jit
    def calculate(uPOD):
        return nu_test * jnp.trapz(uPOD * meanddx, x_domain) - jnp.trapz(uPOD * mean * meandx, x_domain)

    # k - loop
    constant_term = vmap(calculate, in_axes=1)(uPOD)
    return constant_term


@jit
def calculate_linear_term(uPOD, uPODdx, uPODddx):

    @jit
    def calculate(uPOD_k, uPOD_i, uPODdx_i, uPODddx_i):
        return nu_test * jnp.trapz(uPOD_k * uPODddx_i, x_domain) - jnp.trapz(
            uPOD_k * (mean * uPODdx_i + uPOD_i * meandx), x_domain)

    @jit
    def k_loop(uPOD_k, uPOD_i, uPODdx_i, uPODddx_i):
        return vmap(calculate, in_axes=(1, None, None, None))(uPOD_k, uPOD_i, uPODdx_i, uPODddx_i)

    # i - loop
    linear_term = vmap(k_loop, in_axes=(None, 1, 1, 1))(uPOD, uPOD, uPODdx, uPODddx)
    return linear_term


@jit
def calculate_nonlinear_term(uPOD, uPODdx):

    @jit
    def calculate(uPOD_k, uPOD_i, uPODdx):
        return -jnp.trapz(uPOD_k * uPOD_i * uPODdx, x_domain)

    @jit
    def i_loop(uPOD_k, uPOD_i, uPODdx):
        return vmap(calculate, in_axes=(None, 1, None))(uPOD_k, uPOD_i, uPODdx)

    @jit
    def j_loop(uPOD_k, uPOD_i, uPODdx):
        return vmap(i_loop, in_axes=(None, None, 1))(uPOD_k, uPOD_i, uPODdx)

    @jit
    def k_loop(uPOD_k, uPOD_i, uPODdx):
        return vmap(j_loop, in_axes=(1, None, None))(uPOD_k, uPOD_i, uPODdx)

    nonlinear_term = k_loop(uPOD, uPOD, uPODdx)
    return nonlinear_term


constant_term = calculate_constant_term(uPOD)
linear_term = calculate_linear_term(uPOD, uPODdx, uPODddx)
nonlinear_term = calculate_nonlinear_term(uPOD, uPODdx)


@jit
def rhs(galerkin_time_coeffs, t, constant_term, linear_term, nonlinear_term):

    @jit
    def linear_dot(linear_term_k, galerkin_time_coeffs):
        return jnp.dot(linear_term_k, galerkin_time_coeffs)

    r2 = vmap(linear_dot, in_axes=(1, None))(linear_term, galerkin_time_coeffs)

    @jit
    def nonlinear_outer_sum(nonlinear_term_k, galerkin_time_coeffs):
        return jnp.sum(nonlinear_term_k * jnp.outer(galerkin_time_coeffs, galerkin_time_coeffs))

    r3 = vmap(nonlinear_outer_sum, in_axes=(2, None))(nonlinear_term, galerkin_time_coeffs)
    return constant_term + r2 + r3


galerkin_time_coeffs = jode.odeint(rhs, time_coeffs[:, 0], rgk3_time_domain, constant_term, linear_term,
                                   nonlinear_term).transpose()

# ----------------------------------------- FONT ----------------------------------------- #

font = {
    'family': 'serif',
    'name': 'Liberation Serif',
    'color': 'black',
    'weight': 'normal',
    'style': 'italic',
    'size': 20,
}

# ----------------------------------------- EXACT ----------------------------------------- #

# # Plotting exact solutions
# for idx, re in enumerate(re_domain):
#     fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": "3d"})
#     T, X = np.meshgrid(t_domain, x_domain)
#     surf = ax.plot_surface(X,
#                            T,
#                            exact_solution[:, :, idx],
#                            alpha=1,
#                            rstride=1,
#                            cstride=1,
#                            cmap=cm.jet,
#                            linewidth=0,
#                            antialiased=False,
#                            rasterized=True)
#     ax.set_xlabel('x', fontdict=font)
#     ax.set_ylabel('t', fontdict=font)
#     # ax.patch.set_alpha(1.0)
#     ax.axes.set_xlim3d(left=0.02, right=0.98)
#     ax.axes.set_ylim3d(bottom=0.02, top=0.98)
#     ax.set_zlim((0, 0.49))
#     ax.set_zlabel('u(x,t)', labelpad=13, fontdict=font)
#     re = int(re)
#     # ax.set_title(f"Re = {re}", y=0.92, fontdict=font)
#     ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     tmp_planes = ax.zaxis._PLANES
#     ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3], tmp_planes[0], tmp_planes[1], tmp_planes[4], tmp_planes[5])
#     ax.zaxis.set_rotate_label(False)
#     ax.view_init(15, 50, 'z')
#     fig.colorbar(surf, shrink=0.4, aspect=10, location='right')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
#     # plt.savefig(f"figures/Burgers_ExactRe{re}.pdf", bbox_inches='tight', dpi=500)
#     plt.show()

# ----------------------------------------- RIC ----------------------------------------- #

# # Calculate relative importance content
# ric = np.zeros(nTotalSnap)
# total_energy = np.sum(eig_vals)
# acc = 0
# for i in range(len(eig_vals)):
#     acc = acc + eig_vals[i]
#     ric[i] = acc / total_energy * 100

# #      Plot first n_modes important eigenvalues
# figure, axis = plt.subplots()
# axis.plot(np.arange(0, len(eig_vals)), eig_vals, '-o', color='black', ms=5, alpha=1, mfc='red')
# axis.set_xlabel('k', fontdict=font)
# axis.set_ylabel(r'$λ_{k}$', fontdict=font)
# axis.set_xscale('log')
# axis.set_yscale('log')
# axis.set_xlim((1, 1e2))
# axis.grid(which='both')
# axis.annotate(r"$R=6$", (np.arange(0, len(eig_vals))[5], eig_vals[5]),
#               arrowprops={'arrowstyle': '->'},
#               xytext=(-20, -40),
#               textcoords='offset points',
#               fontsize=15)
# plt.tight_layout()
# # plt.savefig('figures/Burgers_Eigenvalues.pdf', dpi=500)
# figure, axis = plt.subplots()
# axis.plot(np.arange(0, len(ric)), ric, '-o', color='black', ms=5, alpha=1, mfc='red')
# axis.set_xlabel('k', fontdict=font)
# axis.set_ylabel(r'$RIC_{k}(\%)$', fontdict=font)
# axis.set_xscale('log')
# axis.set_xlim((1, 1e2))
# axis.grid(which='both')
# axis.annotate(rf"$RIC={ric[5]:0.2f}\%$", (np.arange(0, len(ric))[5], ric[5]),
#               arrowprops={'arrowstyle': '->'},
#               xytext=(-20, -40),
#               textcoords='offset points',
#               fontsize=15)
# plt.tight_layout()
# # plt.savefig("figures/Burgers_RIC.pdf", dpi=500)
# plt.show()

# ----------------------------------------- POD MODES ----------------------------------------- #

# colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']
# for mode_idx in range(uPOD.shape[1]):
#     fig, ax = plt.subplots()
#     ax.set_ylim(-4, 4)
#     ax.grid(which='both')
#     ax.plot(x_domain, uPOD[:, mode_idx], linewidth=2, c=colors[mode_idx])
#     ax.set_xlabel('x', fontdict=font)
#     ax.set_ylabel(rf'$\varphi_{mode_idx+1}(x)$', fontdict=font)
#     # plt.savefig(f'figures/Burgers_PODMode{mode_idx+1}.pdf', dpi=500)
#     plt.show()

# ----------------------------------------- TIME COEFFS ----------------------------------------- #

for i in range(nModes):
    fig, ax = plt.subplots()
    ax.plot(rgk3_time_domain,
            galerkin_time_coeffs[i],
            linestyle='--',
            linewidth=2,
            c='red',
            label='Galerkin Projection')
    ax.plot(rgk3_time_domain, time_coeffs[i], linewidth=2, c='black', label='True')
    ax.set_xlabel('t', fontdict=font)
    ax.set_ylabel(rf'$a_{i+1}(t)$', fontdict=font)
    ax.grid()
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'figures/Burgers_TimeCoeff{i+1}.pdf', dpi=500)
    plt.show()