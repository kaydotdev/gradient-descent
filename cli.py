import numpy as np
import click

from typing import Tuple
from common.alg import SGD, Momentum, NAG, AdaGrad, RMSProp, Adam
from utils.visual import plot_convergence


class FloatListParamType(click.ParamType):
    name = 'float_list'

    def convert(self, value, param, ctx):
        try:
            return [float(item) for item in value.strip("[]").split(",")]
        except ValueError:
            self.fail(f"{value} is not a valid list of floats", param, ctx)


# Instance of the custom parameter type
FLOAT_LIST = FloatListParamType()


@click.command("grad_descent", help="Gradient descent optimization from CLI.")
@click.version_option("1.0.0", prog_name="grad_descent")
@click.argument("epoch", type=click.INT)
@click.argument("function", type=click.STRING)
@click.argument("x", type=FLOAT_LIST)
@click.option("-s", "--step", type=click.FLOAT, default=0.001,
              help='A gradient descent step with the valid range of [0, 1].')
@click.option("-e1", "--eps1", type=click.FLOAT, default=0.001, help='An argumental accuracy parameter.')
@click.option("-e2", "--eps2", type=click.FLOAT, default=0.01, help='A functional accuracy parameter.')
@click.option("-ve", "--var_eps", type=click.FLOAT, default=1e-8,
              help='A gradient component smoothing term. Prevents gradient component G to be 0.')
@click.option("-g", "--gamma", type=click.FLOAT, default=0.9, help='A momentum term [0, 1].')
@click.option("-b1", "--beta1", type=click.FLOAT, default=0.9, help='A weight parameter for the expected value.')
@click.option("-b2", "--beta2", type=click.FLOAT, default=0.999, help='A weight parameter for the unbiased variance.')
@click.option("-h", "--h_gradient", type=click.FLOAT, default=0.0001,
              help='A step of the derivative partitioning grid with the range of [0, 1].')
@click.option("-k", "--order_k", type=click.INT, default=0,
              help='An order of smoothing k. The higher value, the higher gradient precision. If equals 0, '
                   'no smoothing is applied.')
@click.option("-c", "--criteria", default="grad", help='An early stopping criteria.',
              type=click.Choice(["args", "func", "mixed", "grad"], case_sensitive=True))
@click.option("-a", "--alg", default="sgd", help='A gradient descent algorithm.',
              type=click.Choice(["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam"], case_sensitive=True))
@click.option("-m", "--mesh_lim", type=click.Tuple([float, float, float, float]), nargs=4,
              default=(-1.5, 1.5, -1.5, 1.5), help='A render borders for a target function meshgrid in the following '
                                                   'format: (x_min, x_max, y_min, y_max)')
def cli(epoch: int, function: str, x: float, step: float, eps1: float, eps2: float, var_eps: float,
        gamma: float, beta1: float, beta2: float, h_gradient: float, order_k: int, criteria: str,
        alg: str, mesh_lim: Tuple[float, float, float, float]) -> None:
    exp = lambda x: np.exp(x)
    log = lambda x: np.log(x)
    cos = lambda x: np.cos(x)
    sin = lambda x: np.sin(x)
    abs = lambda x: np.abs(x)
    sqrt = lambda x: np.sqrt(x)

    x0 = np.array(x)
    f = eval("lambda x: " + function)
    optimas = {
        "sgd": SGD, "momentum": Momentum, "nag": NAG,
        "adagrad": AdaGrad, "rmsprop": RMSProp, "adam": Adam
    }

    optim_kwargs = dict(step=step, eps1=eps1, eps2=eps2, var_eps=var_eps, h=h_gradient, k=order_k,
                        gamma=gamma, beta1=beta1, beta2=beta2, criteria=criteria)
    x_opt, x_hist, epochs = optimas[alg](f, x0, epoch, **optim_kwargs)

    click.echo(f"Optima element: {x_opt}")
    click.echo(f"Optima function value: {f(x_opt)}")
    click.echo(f"Total iterations spent: {epochs}")

    if len(x0) == 2:
        y_hist = f(x_hist.T)
        plot_convergence(f, (x_opt, x_hist, y_hist, epochs), mesh_lim,
                         f"Algorithm converged in {epochs} iterations")


if __name__ == "__main__":
    cli()
