import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

_color_map = {
    'blue': 'royalblue',
    'green': 'limegreen',
    'magenta': 'violet',
    'yellow': 'gold'
}


def draw_circle(ax, center, color, radius=0.4):
    circ = plt.Circle(center, radius, fc=to_rgba(color, 0.8), ec=color)
    ax.add_patch(circ)


def draw_zones(ax, zone_positions):
    for zone in zone_positions:
        color = zone.split('_')[0]
        if color in _color_map:
            color = _color_map[color]
        draw_circle(ax, zone_positions[zone], color)


def draw_path(ax, points, color, linewidth=2, markersize=5, draw_markers=False):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    ax.plot(x, y, color=color, linestyle='solid', marker=None, linewidth=linewidth)
    if draw_markers:
        for i in range(20, len(points), 20):
            ax.plot(x[i], y[i], marker='o', color=color, markersize=markersize)
        ax.plot(x[-1], y[-1], marker='o', color=color, markersize=markersize)


def draw_diamond(ax, center, color, size=0.12):
    diamond_shape = plt.Polygon(
        [(center[0], center[1] + size),
         (center[0] + size, center[1]),
         (center[0], center[1] - size),
         (center[0] - size, center[1])],
        color=color,
        zorder=10
    )
    ax.add_patch(diamond_shape)


def draw_trajectories(zone_positions, paths, num_cols, num_rows):
    if len(zone_positions) != len(paths):
        raise ValueError('Number of zone positions and paths must be the same')
    if num_cols * num_rows < len(zone_positions):
        raise ValueError('Number of zone positions exceeds the number of subplots')
    fig = plt.figure(figsize=(20, 5))
    for i, (zone_poss, path) in enumerate(zip(zone_positions, paths)):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        draw_trajectory(ax, zone_poss, path)
    plt.tight_layout(pad=2)
    return fig


def draw_trajectory(ax, zone_positions, path):
    setup_axis(ax)
    draw_zones(ax, zone_positions)
    draw_path(ax, path, color='green')
    draw_diamond(ax, path[0], color='orange')


def setup_axis(ax):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    ax.grid(True, which='both', color='gray', linestyle='dashed', linewidth=1, alpha=0.5)

    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_yticks(np.arange(-3, 4, 1))

    # Add border
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1)
