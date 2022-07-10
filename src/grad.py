import numpy as np

from typing import Callable


def grad_left(F: Callable[[np.array], np.array], x: np.array, h=0.001) -> np.array:
    """A finite-difference approximation for left-side gradient $\nabla F_{-}(x)$ with the precision order $O(h^2)$.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): an input vector $x \in \mathbb{R}^n$, where the derivative is calculated.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.

    Returns:
        np.array: a gradient vector approximation $\nabla F_{-}(x)$.
    """

    n, grad = len(x), np.zeros(x.shape)
    
    for i in range(n):
        vh = h * np.eye(1, n, i).reshape((n, ))
        grad[i] = (-3.0 * F(x) + 4.0 * F(x + vh) - F(x + 2.0 * vh)) / (2.0 * h)
    
    return grad


def grad_center(F: Callable[[np.array], np.array], x: np.array, h=0.001) -> np.array:
    """A finite-difference approximation for central gradient $\nabla F(x)$ with the precision order $O(h^2)$.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): an input vector $x \in \mathbb{R}^n$, where the derivative is calculated.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.

    Returns:
        np.array: a gradient vector approximation $\nabla F(x)$.
    """

    n, grad = len(x), np.zeros(x.shape)
    
    for i in range(n):
        vh = h * np.eye(1, n, i).reshape((n, ))
        grad[i] = (F(x + vh) - F(x - vh)) / (2.0 * h)
    
    return grad


def grad_right(F: Callable[[np.array], np.array], x: np.array, h=0.001) -> np.array:
    """A finite-difference approximation for right-side gradient $\nabla F_{+}(x)$ with the precision order $O(h^2)$.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$.
        x (np.array): an input vector $x \in \mathbb{R}^n$, where the derivative is calculated.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.

    Returns:
        np.array: a gradient vector approximation $\nabla F_{+}(x)$.
    """

    n, grad = len(x), np.zeros(x.shape)

    for i in range(n):
        vh = h * np.eye(1, n, i).reshape((n, ))
        grad[i] = (F(x - 2.0 * vh) - 4.0 * F(x - vh) + 3.0 * F(x)) / (2.0 * h)

    return grad


def grad_smoothed(F: Callable[[np.array], np.array], x: np.array, h=0.001, k=10) -> np.array:
    """A smoothed finite-difference approximation for gradient $\nabla F_{h}(x)$ with the precision order $O(h^2)$. 
Can be also applied to the non-smooth and (or) non-convex target function.

    Args:
        F (Callable[[np.array], np.array]): a target function $F(x)$ with a single input argument $x \in \mathbb{R}^n$. 
        x (np.array): an input vector $x \in \mathbb{R}^n$, where the derivative is calculated.
        h (float, optional): a step of the derivative partitioning grid with the range of $0<h<1$. The lower value, the higher gradient precision. Defaults to 0.001.
        k (int, optional): a order of smoothing $k \in \mathbb{N}$. The higher value, the higher gradient precision. Defaults to 10.

    Returns:
        np.array: a gradient vector approximation $\nabla F_{h}(x)$.
    """

    n, grad = len(x), np.zeros(x.shape)

    for _ in range(k):
        grad_component = np.zeros(x.shape)
        dir = np.random.uniform(0.0, 1.0, (1, n)).reshape((n, ))
        yi = dir / np.linalg.norm(dir)

        for j in range(n):
            vh = h * np.eye(1, n, j).reshape((n, ))
            grad_component[j] = (F(x + vh * yi) - F(x - vh * yi)) / (2.0 * h)
        
        grad += grad_component
    
    return grad / k
