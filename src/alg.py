import numpy as np

from tqdm import tqdm
from typing import Callable, Tuple, Literal
from grad import grad_center, grad_smoothed


CriteriaType = Literal["args", "func", "mixed", "grad"]
"""A stop criteria for a gradient descent algorithms:
 - "args" - norm of difference between adjacent element: $|| x^{i} - x^{i-1} || < \varepsilon_{1};$
 - "func" - norm of difference between fucntions value of adjacent element: $|| F(x^{i}) - F(x^{i-1}) || < \varepsilon_{2};$
 - "mixed" - combined criterias of "args" and "func": $|| x^{i} - x^{i-1} || < \varepsilon_{1}, || F_{h}(x^{i}) - F_{h}(x^{i-1}) || < \varepsilon_{2};$
 - "grad" - norm of a gradient value in the current element: || \nabla F(x^{i}) || < \varepsilon_{1}.$$
"""


def SGD(F: Callable[[np.array], np.array], x: np.array, epoch: int,
        step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """A Stochastic Gradient Descent (SGD) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        x = x - step * grad
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def Momentum(F: Callable[[np.array], np.array], x: np.array, epoch: int, gamma: float,
             step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """A Momentum Gradient Descent (Momentum GD) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        gamma (float): a momentum term $0< \gamma < 1$.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    vt = np.zeros(x.shape)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        v = gamma * vt + step * grad
        x, vt = x - v, v
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def NAG(F: Callable[[np.array], np.array], x: np.array, epoch: int, gamma: float,
        step=0.001, eps1=0.001, eps2=0.01, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """A Nesterov Accelerated Gradient (NAG) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        gamma (float): a momentum term $0< \gamma < 1$.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    vt = np.zeros(x.shape)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x - gamma * vt, h) if k == 0 else grad_smoothed(F, x - gamma * vt, h, k)
        v = gamma * vt + step * grad
        x, vt = x - v, v
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def AdaGrad(F: Callable[[np.array], np.array], x: np.array, epoch: int,
            step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """An Adaptive gradient (Adagrad) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        vareps (float, optional): a gradient component smoothing term. Prevents gradient component $G$ to be 0. Defaults to 1e-8.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    G = np.zeros(x.shape)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        G = G + grad ** 2
        x = x - (step / np.sqrt(G + vareps)) * grad
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def RMSProp(F: Callable[[np.array], np.array], x: np.array, epoch: int,
            step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, beta=0.9, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """A Root-Mean Squared Propagation (RMSProp) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        vareps (float, optional): a gradient component smoothing term. Prevents gradient component $G$ to be 0. Defaults to 1e-8.
        beta (float, optional): a weight parameter for the unbiased variance. Defaults to 0.9.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    G = np.zeros(x.shape)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)
        G = beta * G + (1 - beta) * grad ** 2
        x = x - (step / np.sqrt(G + vareps)) * grad
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter


def Adam(F: Callable[[np.array], np.array], x: np.array, epoch: int,
         step=0.001, eps1=0.001, eps2=0.01, vareps=1e-8, beta1=0.9, beta2=0.999, h=0.0001, k=0, criteria="grad") -> Tuple[np.array, np.array, int]:
    """An Adaptive Moment Estimations (ADAM) algorithm.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): a start element $x_0$ in a sequence.
        epoch (int): a maximum number of iterations.
        step (float, optional): a gradient descent step with the valid range of $0< \lambda <1$. Defaults to 0.001.
        eps1 (float, optional): an argumental accuracy parameter. Defaults to 0.001.
        eps2 (float, optional): a functional accuracy parameter. Defaults to 0.01.
        vareps (float, optional): a gradient component smoothing term. Prevents gradient component $G$ to be 0. Defaults to 1e-8
        beta1 (float, optional): a weight parameter for the expected value. Defaults to 0.9.
        beta2 (float, optional): a weight parameter for the unbiased variance. Defaults to 0.999.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. If equals 0, no smoothing is applied. Defaults to 0.
        criteria (str, optional): an early stopping criteria. Defaults to "grad".

    Returns:
        Tuple[np.array, np.array, int]: An optimal vector, a gradient descent sequence, and a number of iterations.
    """

    trange = tqdm(range(epoch))
    xi = np.copy(x)
    m = np.zeros(x.shape)
    v = np.zeros(x.shape)
    iter = 0

    for _ in trange:
        grad = grad_center(F, x, h) if k == 0 else grad_smoothed(F, x, h, k)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        x = x - (step / np.sqrt(v + vareps)) * m
        xi = np.vstack((xi, x))

        trange.set_postfix({ "x": np.round(x, 6), "F": np.round(F(x), 6) })

        if criteria == "args" and np.linalg.norm(xi[-1] - xi[-2]) < eps1: break
        elif criteria == "func" and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2: break
        elif criteria == "mixed" and (np.linalg.norm(xi[-1] - xi[-2]) < eps1 and np.linalg.norm(F(xi[-1]) - F(xi[-2])) < eps2): break
        elif criteria == "grad" and np.linalg.norm(grad) < eps1: break
        else: iter += 1

    return x, xi, iter
