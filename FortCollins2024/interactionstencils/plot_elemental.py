#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)

ax.xaxis.set_major_locator(MultipleLocator(1.000))
ax.yaxis.set_major_locator(MultipleLocator(1.000))

n_x_cells = 6
n_y_cells = n_x_cells
kernel_radius = 1

ax.set_xlim(0 - kernel_radius, n_x_cells + kernel_radius)
ax.set_ylim(0 - kernel_radius, n_y_cells + kernel_radius)

ax.tick_params(which="major", labelleft=False, labelbottom=False, left=False, bottom=False)

ax.grid(linestyle="-", linewidth=1.0, color="black")

def circle(x, y, radius=0.15):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke

    circle = Circle(
        (x, y),
        radius,
        clip_on=False,
        zorder=10,
        linewidth=1,
        edgecolor="black",
        facecolor=(0, 0, 0, 0),
        path_effects=[withStroke(linewidth=5, foreground="w")],
    )
    ax.add_artist(circle)

def triangle(points):
    from matplotlib.patches import Polygon

    assert points.shape == (3, 2)

    tri = Polygon(
        points,
        clip_on=False,
        alpha=0.5,
        zorder=20,
        linewidth=1,
        edgecolor="black",
        facecolor="red",
    )
    ax.add_artist(tri)

def gauss_points(points):
    from matplotlib.patches import Rectangle

    assert points.shape == (3, 2)

    # not a real Gauss rule but it looks nice
    reference_points = np.array([[0.3333333, 0.3333333],
                                 [0.70000000000000000000, 0.10000000000000000000],
                                 [0.10000000000000000000, 0.70000000000000000000],
                                 [0.10000000000000000000, 0.10000000000000000000]])

    B = np.array([[points[1][0] - points[0][0], points[2][0] - points[0][0]],
                  [points[1][1] - points[0][1], points[2][1] - points[0][1]]])

    physical_points = (B @ reference_points.T).T
    physical_points[:, 0] += points[0][0]
    physical_points[:, 1] += points[0][1]

    for interaction_point in physical_points:
        x_left = np.round(interaction_point[0] - 0.5) + 0.5
        y_bottom = np.round(interaction_point[1] - 0.5) + 0.5

        rect = Rectangle((x_left - kernel_radius, y_bottom - kernel_radius), 2*kernel_radius, 2*kernel_radius, alpha=0.20, zorder=25, linewidth=4, edgecolor="black", facecolor="blue")
        ax.add_artist(rect)

    ax.plot(physical_points[:, 0], physical_points[:, 1], 'X', zorder=30, color='black', linewidth=3)

# plot cell-centered data
for i in range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1])):
    for j in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])):
        circle(i + 0.5, j + 0.5, radius=0.1)

t1_vertices = np.array([[-0.2, 1.6], [5.4, 2.4], [1.6, 5.3]])*1.25
triangle(t1_vertices)

gauss_points(t1_vertices)

t2_vertices = np.array([[-0.2, 1.6], [5.4, 2.4], [3.6, -0.1]])*1.25
triangle(t2_vertices)

gauss_points(t2_vertices)

plt.savefig("elemental_stencil.pdf", bbox_inches='tight')
plt.savefig("elemental_stencil.png", bbox_inches='tight')
