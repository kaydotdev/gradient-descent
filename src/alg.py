import numpy as np

from typing import Callable
from grad import grad_center, grad_smoothed


CriteriaType = "args" | "func" | "mixed" | "grad";
"""A stop criteria for a gradient descent algorithms:
 - "args" - Norm of difference between adjacent element: $| x^{i} - x^{i-1} | < \varepsilon_{1}$
 - "func" - Norm of difference between fucntions value of adjacent element: $| F_{h}(x^{i}) - F_{h}(x^{i-1}) | < \varepsilon_{2}$
 - "mixed" - Combined criterias of "args" and "func": $| x^{i} - x^{i-1} | < \varepsilon_{1}, | F_{h}(x^{i}) - F_{h}(x^{i-1}) | < \varepsilon_{2}$
 - "grad" - Norm of a gradient value in the current element: $$
"""


def SGD(F: Callable[[np.array], np.array], x: np.array, epoch: int,
        step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        x = x - step * grad
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def Momentum(F: Callable[[np.array], np.array], x: np.array, epoch: int, gamma: float,
             step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    vt = np.zeros(x.shape)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        v = gamma * vt + step * grad
        x, vt = x - v, v
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def NAG(F: Callable[[np.array], np.array], x: np.array, epoch: int, gamma: float,
        step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    vt = np.zeros(x.shape)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x - gamma * vt, h) if k == 0 else grad_smoothed(F, x - gamma * vt, h, k)
        v = gamma * vt + step * grad
        x, vt = x - v, v
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def AdaGrad(F: Callable[[np.array], np.array], x: np.array, epoch: int,
            step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    G = np.zeros(x.shape)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        G = G + grad ** 2
        x = x - (step / np.sqrt(G + vareps)) * grad
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def RMSProp(F: Callable[[np.array], np.array], x: np.array, epoch: int,
            step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, beta=0.9, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    G = np.zeros(x.shape)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        G = beta * G + (1 - beta) * grad ** 2
        x = x - (step / np.sqrt(G + vareps)) * grad
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def Adam(F: Callable[[np.array], np.array], x: np.array, epoch: int,
         step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, beta1=0.9, beta2=0.999, h=0.0001, k=0, criteria="grad"):
    xi = np.copy(x)
    m = np.zeros(x.shape)
    v = np.zeros(x.shape)
    iter = 0

    for _ in range(epoch):
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        x = x - (step / np.sqrt(v + vareps)) * m
        xi = np.vstack((xi, x))

        if np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter
