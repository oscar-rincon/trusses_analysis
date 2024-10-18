# -*- coding: utf-8 -*-
"""
Visualize the topography of the Aburra Valley using
shading and a personalized colormap.

@author: Nicolás Guarín-Zapata
@date: October 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LightSource


def set_axes_equal(ax, scaling=0.6):
    """Make axes of 3D plot have equal scale.

    Make axes of 3D plot have equal scale so that spheres
    appear as spheres,     cubes as cubes, etc..  This is one
    possible solution to Matplotlib's ax.set_aspect('equal')
    and ax.axis('equal') not working for 3D.
    
    Taken from: https://stackoverflow.com/a/31364297/3358223

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = scaling * 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



# Datos
pts = np.loadtxt("valle_aburra-quads.pts", skiprows=2)
x, y, z = pts.T
x.shape = 272, 173
y.shape = 272, 173
z.shape = 272, 173
y = y[::-1, :]

# Derivadas
dx = x[0, 1]  - x[0, 0]
dzdx, dzdy = np.gradient(z, dx)
grad_mag = np.sqrt(dzdx**2 + dzdy**2)

angulo = np.arctan(grad_mag) * 180/np.pi


#%% Visualización
colores = ["#63b82a", "#d1d26a", "#ffe8a5", "#fb9549", "#ef7272"]
norm = plt.Normalize(0, 90)
cmap = LinearSegmentedColormap.from_list("", colores)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ls = LightSource(270, 45)
rgb = ls.shade(3*z, cmap=cmap, vert_exag=0.1, blend_mode="soft")
surf = ax.plot_surface(x, y, 3*z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=True)
set_axes_equal(ax)
ax.axis("off")
 


