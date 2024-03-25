import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from typing import Callable, Tuple


def build_meshgrid(borders: Tuple, grid_density=0.01):
    x_min, x_max, y_min, y_max = borders
    return np.meshgrid(np.arange(x_min, x_max, grid_density), np.arange(y_min, y_max, grid_density))


def plot_convergence(F: Callable[[np.array], np.array], paths: Tuple, borders: Tuple, label: str, grid_density=0.01):
    x_opt, x_hist, y_hist, it = paths
    x_grid = build_meshgrid(borders, grid_density=grid_density)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 2, (1, 3), projection='3d')

    fig.colorbar(ax.contour3D(x_grid[0], x_grid[1], F(x_grid), 50, cmap=cm.coolwarm, antialiased=True))
    ax.plot(x_hist.T[0], x_hist.T[1], y_hist, label=label)
    ax.grid()

    ax.set_title(f"Convergence path with optima in {np.around(x_opt, 2)}", fontsize="small")
    ax.legend(fontsize="small")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_hist.T[0], x_hist.T[1], label=label, marker=".")
    ax.grid()

    ax.set_title("The projection of a convergence path")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(range(y_hist.shape[0]), y_hist, label=label, marker=".")
    ax.grid()

    ax.set_title(f"Target function value per iteration. Total iterations: {it}")
    ax.legend()

    fig.tight_layout()

    plt.show()
