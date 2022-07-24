import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from typing import Callable, Tuple


def plot_stats(F: Callable[[np.array], np.array], pathes: Tuple, borders: Tuple, label: str, grid_dens=0.01):
    x_opt, x_hist, y_hist, it = pathes
    xmin, xmax, ymin, ymax = borders
    X = np.meshgrid(np.arange(xmin, xmax, grid_dens), np.arange(ymin, ymax, grid_dens))

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 2, (1, 3), projection='3d')

    fig.colorbar(ax.contour3D(X[0], X[1], F(X), 50, cmap=cm.coolwarm, antialiased=True))
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
