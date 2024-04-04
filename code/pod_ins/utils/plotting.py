import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy.interpolate import griddata


def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def plot_square_data(x, y, data, resolution=1000, cylinder=(-0.5, 0.5, 1, 1), **kwargs):
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (4, 4)))
    xmin = np.round(np.min(x), 1)
    xmax = np.round(np.max(x), 1)
    ymin = np.round(np.min(y), 1)
    ymax = np.round(np.max(y), 1)
    far = max(map(abs, (xmin, xmax, ymin, ymax)))
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # interpolation grid
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution
    gridx = np.arange(xmin, xmax, dx)
    gridy = np.arange(ymin, ymax, dy)
    gridx, gridy = np.meshgrid(gridx, gridy)
    # interpolate data
    grid = griddata(np.column_stack((x, y)), data, (gridx, gridy))
    cntr = ax.contourf(gridx, gridy, grid, levels=100, cmap='jet')
    sqr_cylinder = patches.Rectangle((-0.5, -0.5), cylinder[2], cylinder[3], color='white')
    ax.set_aspect(1.0)
    ax.add_patch(sqr_cylinder)
    return gridx, gridy, grid


# plotting cylinder data
def plot_cylinder_data(x, y, data, levels=100, cmap='jet', ax=None, zoom=False, resolution=1000, cbar=True):
    # construct plotting axis if necessary
    if ax == None:
        if zoom:
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    far = np.round(np.max(y), 1)

    # plot central cylinder and farfield
    cylinder = plt.Circle((0, 0), radius=0.5, color="white")
    farfield = plt.Circle((0, 0), radius=far, color="magenta", fill=False)

    # set axis limits
    if zoom:
        xlim_l = -far * 0.2
        xlim_r = far * 0.8
        ylim_r = far / 5
        ylim_l = -far / 5
    else:
        xlim_l = -far
        xlim_r = far
        ylim_r = far
        ylim_l = -far

    ax.set_xlim([xlim_l, xlim_r])
    ax.set_ylim([ylim_l, ylim_r])

    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # distance function
    def dist(xm, ym, x, y):
        return np.sqrt((x - xm) * (x - xm) + (y - ym) * (y - ym))

    # add equidistant grid points
    gridx = np.arange(xlim_l, xlim_r, (xlim_r - xlim_l) / resolution)
    gridy = np.arange(ylim_l, ylim_r, (ylim_r - ylim_l) / resolution)
    grid_x, grid_y = np.meshgrid(gridx, gridy)
    dx = (xlim_r - xlim_l) / resolution
    dy = (ylim_r - ylim_l) / resolution
    points = np.stack((x, y), axis=1)

    # interpolate data
    grid = griddata(points, data, (grid_x, grid_y))

    # set boundary conditions
    r = 0.5
    far = far
    if zoom:
        xm = int(resolution * 0.2)
        ym = int(resolution / 2)
    else:
        xm = int(resolution / 2)
        ym = int(resolution / 2)

    for i in range(xm - int(r / dx), xm + int(r / dx), 1):
        for j in range(ym - int(r / dy), ym + int(r / dy), 1):
            if dist(xm, ym, i, j) < (r / dx):
                grid[j, i] = 0

    for i in range(resolution):
        for j in range(resolution):
            if dist(xm, ym, i, j) > (far / dx):
                grid[i, j] = 0

    cntr = ax.contourf(grid_x, grid_y, grid, levels, cmap=cmap)
    ax.add_patch(cylinder)
    ax.add_patch(farfield)

    # add colorbar
    if cbar:
        plt.colorbar(cntr, ax=ax)
