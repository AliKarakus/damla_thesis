from telnetlib import GA
from matplotlib.rcsetup import cycler
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import arange, linalg as LA
from scipy.integrate import simps
from scipy import integrate
import math as m


def galerkin_coefficients(W, X, Y, uMean, vMean, uPOD, vPOD):
# q_ijk = - Int dxdy [ ( u_j*d/dx(u_k) + v_j*d/dy(u_k) )*u_i + ...
# ( u_j*d/dx(v_k) + v_j*d/dy(v_k) )*v_i]
    nModes = uPOD.shape[0]
    nx = uMean.shape[1]
    ny = uMean.shape[0]
    dx = X[0][1]-X[0][0]
    dy = Y[1][0]-Y[0][0]

    GalerkinCoeff = np.zeros((nModes+1,nModes,nModes+1))

    dukdx = np.zeros((ny,nx), dtype=np.float128) 
    dukdy = np.zeros((ny,nx),dtype=np.float128) 
    dvkdx = np.zeros((ny,nx),dtype=np.float128) 
    dvkdy = np.zeros((ny,nx),dtype=np.float128) 

    u = np.zeros((nModes+1,ny,nx),dtype=np.float128) 
    v = np.zeros((nModes+1,ny,nx),dtype=np.float128)
    u[0, :, :] = uMean
    v[0, :, :] = vMean
    u[1:nModes+1, :, :] = uPOD
    v[1:nModes+1, :, :] = vPOD

    for i in range(nModes):
        for j in range(nModes+1):
            for k in range(nModes+1):
                dukdx, dukdy = np.gradient(u[k, :, :], dx, axis=1), np.gradient(u[k, :, :], dy, axis=0)
                dvkdx, dvkdy = np.gradient(v[k, :, :],dx, axis=1), np.gradient(v[k, :, :],dy, axis=0)
                GalerkinCoeff[k, i, j] = -np.sum(np.sum(W * ((u[j, :, :] * dukdx + v[j, :, :] * dukdy) * u[i+1, :, :] +
                                             (u[j, :, :] * dvkdx + v[j, :, :] * dvkdy) * v[i+1, :, :] )))

    return GalerkinCoeff

def stuart_gs(tInt, TimeCoeffGalerkin):
    global GalerkinCoeff
    nModes = len(TimeCoeffGalerkin)
    TimeCoeffGalerkinInsert = np.insert(TimeCoeffGalerkin, 0, 1)
    dadt = np.zeros(nModes)
    for i in range(nModes):
        for j in range(nModes+1):
            for k in range(nModes+1):
                dadt[i] = dadt[i] + GalerkinCoeff[k, i, j] * TimeCoeffGalerkinInsert[j] * TimeCoeffGalerkinInsert[k]
    return dadt


# Input Variables
C = 1.1                # C=1: laminar shear layer, C>1: cat eyes
A = m.sqrt(C**2 - 1)   # A = sqrt(C^2-1)
e = A / C              # Vortex Strength

c  = 1          # convection velocity
nx = 100        # number of grid points in x-direction
ny = 50         # number of grid points in y-direction
nt = 50         # number of snapshots 
nModes = 10     # Number of modes

T  = 2 * m.pi               # Final time
dt = T / (nt-1)             # Time Step Size
t  = np.linspace(0, T, nt)  # Time domain

eps1 = 1e-10   # tolerance for orthonormality
eps2 = 1e-6    # tolerance for Runge Kutta GS integration

Lx = 4 * m.pi  # Domain length in x direction - two wave lengths
Ly = 6         # Domain length in y direction - +-3
dx = Lx/(nx-1) 
dy = Ly/(ny-1) 

x = np.linspace(-Lx/2, Lx/2, nx)    # x - domain
y = np.linspace(-Ly/2, Ly/2, ny)    # y - domain
X, Y = np.meshgrid(x, y)            # Grid

u = np.zeros((nt, ny, nx))
v = np.zeros((nt, ny, nx))

for i in range(nt):
    
    u[i,:,:] = c + np.sinh(Y) / ( np.cosh(Y) + e*np.cos(X-c*t[i]))
    v[i,:,:] = e*np.sin(X-c*t[i]) / (np.cosh(Y) + e*np.cos(X-c*t[i]))

uMean = np.mean(u, axis=0)
vMean = np.mean(v, axis=0)

uFluc = np.zeros((nt, ny, nx))
vFluc = np.zeros((nt, ny, nx))

for i in range(nt):
    uFluc[i,:,:] = u[i,:,:] - uMean
    vFluc[i,:,:] = v[i,:,:] - vMean

W = dx * dy *  np.ones((ny, nx))

W[:, 0] = 0.5*dx*dy
W[:, -1] = 0.5*dx*dy
W[0, :]  = 0.5*dx*dy
W[-1, :] = 0.5*dx*dy

W[0 , 0] = 0.25*dx*dy
W[-1, 0] = 0.25*dx*dy
W[-1,-1] = 0.25*dx*dy
W[0 ,-1] = 0.25*dx*dy

# correlation1 = np.zeros((nt, nt))
# for i in range(nt):
#     for j in range(nt):
#         correlation1[i][j] = np.sum(np.sum(W * (uFluc[i, :, :] * uFluc[j, :, :] +
#                                                vFluc[i, :, :] * vFluc[j, :, :])))
#         correlation1[j][i] = correlation1[i][j]
# correlation1 = correlation1 / nt

correlation = np.zeros((nt, nt))
for i in range(nt):
    for j in range(nt):
        A = uFluc[i, :, :] * uFluc[j, :, :] + vFluc[i, :, :] * vFluc[j, :, :]
        correlation[i][j] = simps(simps(uFluc[i, :, :] * uFluc[j, :, :] +
                                                vFluc[i, :, :] * vFluc[j, :, :], x), y)
        correlation[j][i] = correlation[i][j]
correlation = correlation / nt

# Eigenvalues and eigenvector of correlation matrix.
eig_vals, eig_vecs = LA.eig(correlation)
eig_vals = np.real(eig_vals)
eig_vecs = np.real(eig_vecs)

# Sort eigenvalues and vectors according to abs(eig_val)
sorting_indices = np.abs(eig_vals).argsort()[::-1]
eig_vals = eig_vals[sorting_indices]
eig_vecs = eig_vecs[:, sorting_indices]

# Scale Time Coefficients
TimeCoeff = np.zeros((nt, nModes))

for i in range(nModes):
    for j in range(nt):
        ModeFactor = np.sqrt(nt*eig_vals[i])
        TimeCoeff[j][i] = eig_vecs[j][i]*ModeFactor

# Compute POD modes
uPOD = np.zeros((nModes, ny, nx ))
vPOD = np.zeros((nModes, ny, nx ))

for i in range(nModes):
    for j in range(nt):
        uPOD[i, :, :] = uPOD[i, :, :] + eig_vecs[j][i] * uFluc[j, :, :]
        vPOD[i, :, :] = vPOD[i, :, :] + eig_vecs[j][i] * vFluc[j, :, :]
    
    modeFactor = 1 / np.sqrt(nt*eig_vals[i])
    uPOD[i, :, :] = uPOD[i, :, :]*modeFactor
    vPOD[i, :, :] = vPOD[i, :, :]*modeFactor

# Check normalization of POD modes
Id = np.zeros((nModes,nModes))
for i in range(nModes):
    for j in range(nModes):
        Id[i][j] = np.sum(np.sum(W * (uPOD[i, :, :] * uPOD[j, :, :] +
                                      vPOD[i, :, :] * vPOD[j, :, :])))

if np.abs(LA.norm(np.eye(nModes) - Id)) > eps1:
    print('POD modes not orthonormal')

# Compute Time Coefficients by projection (Check)
TimeCoeffProj = np.zeros((nt, nModes))

for i in range(nModes):
    for j in range(nt):
        TimeCoeffProj[j][i] = np.sum(np.sum(W * (uFluc[j, :, :] * uPOD[i, :, :] +
                                                vFluc[j, :, :] * vPOD[i, :, :])))

global GalerkinCoeff
GalerkinCoeff = galerkin_coefficients(W, X, Y, uMean, vMean, uPOD, vPOD)

t0, t1 = 0, 4*m.pi                                              # start and end
tInt = np.linspace(t0, t1, 100)                                 # the points of evaluation of solution
TimeCoeffIC = TimeCoeffProj[0, :]                               # initial value
TimeCoeffGalerkin = np.zeros((len(tInt), len(TimeCoeffIC)))     # array for solution

TimeCoeffGalerkin[0, :] = TimeCoeffIC

r = integrate.ode(stuart_gs).set_integrator("dopri5")        # choice of method
r.set_initial_value(TimeCoeffIC, t0)                         # initial values
for i in range(1, tInt.size):
   TimeCoeffGalerkin[i, :] = r.integrate(tInt[i])            # get one more value, add it to the array
   if not r.successful():
       raise RuntimeError("Could not integrate")

u_field = np.zeros((nt, ny, nx))
v_field = np.zeros((nt, ny, nx))

for time in range(nt):
    for mode in range(nModes):
        u_field[time, :, :] += TimeCoeff[time, mode] * uPOD[mode, :, :]
        v_field[time, :, :] += TimeCoeff[time, mode] * vPOD[mode, :, :]
    u_field[time, :, :] += uMean
    v_field[time, :, :] += vMean

# ----------------------------------------- FIGURES ----------------------------------------- #

# Plot eigenvalues and relative information content. 

font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 12,
}

relative_importance_content = np.zeros(nt)
total_energy = np.sum(eig_vals)
acc = 0
for i in range(len(eig_vals)):
  acc = acc + eig_vals[i]
  relative_importance_content[i] = acc / total_energy * 100

# Plot comparison projected and GS integrated Fourier coefficients
# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
lines = []
for mode in range(2):
    l, = plt.plot(t, TimeCoeffProj[:, mode], linestyle='-', marker='+', linewidth=4, markersize=10, alpha=0.8, label=rf'$a_{mode}{{t}}$ - True')
    l2, = plt.plot(tInt, TimeCoeffGalerkin[:, mode], linestyle='--', marker='o', linewidth=4, markersize=5, alpha=0.8, label=rf'$a_{mode}{{t}}$ - GP')
    plt.xlabel("Time", fontdict=font)
    plt.ylabel("Time coefficients", fontdict=font)
    lines.append(l)
    lines.append(l2)
plt.legend(handles=lines)
plt.show()

# Plot comparison real and reconstructed vorticity fields
for time in range(nt):
    print(f'{time=}', end='\r')
    dukdy = np.gradient(u_field[time, :, :], dy, axis=0)
    dvkdx = np.gradient(v_field[time, :, :], dx, axis=1)
    vorticityS = dvkdx - dukdy
    
    fig, (axl, axr) = plt.subplots(2, 1, figsize=(8,8), dpi=150)
    lcontour = axl.contourf(X, Y, vorticityS, 100, cmap='turbo')
    axl.set_xlabel('x', fontdict=font)
    axl.set_ylabel('y', fontdict=font)
    ymin,ymax = axl.get_ylim()
    xmin,xmax = axl.get_xlim()
    aspect = (ymax-ymin)/(xmax-xmin)
    axl.set_aspect('equal')
    fig.colorbar(lcontour, ax=axl, fraction=0.046*aspect, pad=0.04)
    axl.set_title(f'Reconstructed vorticity from POD modes at time = {time}', fontdict=font)

    # Plot vorticity and velocity vectors at time = 1
    dukdy = np.gradient(u[time, :, :], dy, axis=0)
    dvkdx = np.gradient(v[time, :, :], dx, axis=1)
    vorticity = dvkdx - dukdy
    rcontour = axr.contourf(X, Y, vorticity, 100, cmap='turbo')
    axr.set_xlabel('x', fontdict=font)
    axr.set_ylabel('y', fontdict=font)
    ymin,ymax = axr.get_ylim()
    xmin,xmax = axr.get_xlim()
    aspect = (ymax-ymin)/(xmax-xmin)
    axr.set_aspect('equal')
    fig.colorbar(rcontour, ax=axr, fraction=0.046*aspect, pad=0.04)
    axr.set_title(f'Vorticity at time = {time}', fontdict=font)
    plt.tight_layout()
    plt.savefig(f"figs/vorticity_time{time:05d}.png")
    axl.quiver(X, Y, u_field[-1, :, :], v_field[-1, :, :], scale_units='xy', scale=5)
    axr.quiver(X, Y,u[-1, :, :], v[-1, :, :], scale_units='xy', scale=5)
    plt.savefig(f"figs_quiver/vorticity_quiver_time{time:05d}.png")
    plt.close()

#       Plot first n_modes important eigenvalues
figure, axis = plt.subplots(1, 2)
axis[0].plot(np.arange(1, nModes + 1), eig_vals[:nModes], '-o', color='black', ms=8, alpha=1, mfc='red')
axis[0].set_xlabel('k', fontdict=font)
axis[0].set_ylabel(r'$λ_{k}$', fontdict=font)
axis[0].grid()

axis[1].plot(np.arange(1, nModes + 1),
             relative_importance_content[:nModes],
             '-o',
             color='black',
             ms=8,
             alpha=1,
             mfc='red')
axis[1].set_xlabel('k', fontdict=font)
axis[1].set_ylabel(r'$RIC_{k}(\%)$', fontdict=font)
axis[1].grid()
plt.show()

#       Plot time coefficients
for i in range(0,nModes,2):
    plt.plot(t,TimeCoeff[:,i],'b-', label=r'$a_{}$'.format(i))
    plt.plot(t,TimeCoeff[:,i+1],'r-', label=r'$a_{}$'.format(i+1))
    plt.xlabel('t', fontdict=font)
    plt.ylabel(r'$a_{i}$', fontdict=font)
    plt.suptitle('Time coefficients')
    plt.legend()
    plt.grid()
    plt.show()

#  Plot POD modes
for i in range(nModes):

    dukdy = np.gradient(uPOD[i, :, :], dy, axis=0)
    dvkdx = np.gradient(vPOD[i, :, :], dx, axis=1)
    vorticity = dvkdx - dukdy

    plt.contourf(X, Y, vorticity, 100, cmap='turbo')
    plt.colorbar()
    plt.quiver(X[::2], Y[::2], uPOD[i, ::2], vPOD[i, ::2], scale_units='xy', scale=1)
    plt.xlabel('x', fontdict=font)
    plt.ylabel('y', fontdict=font)
    plt.suptitle('POD Mode {}'.format(i+1), fontdict=font)
    plt.show()


















