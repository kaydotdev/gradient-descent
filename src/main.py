import argparse
import numpy as np

from alg import *


exp = lambda x: np.exp(x)
log = lambda x: np.log(x)
cos = lambda x: np.cos(x)
sin = lambda x: np.sin(x)
abs = lambda x: np.abs(x)
sqrt = lambda x: np.sqrt(x)

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("epoch", type=int, help="a maximum number of iterations.")
    parser.add_argument("function", type=str, help="a target function F(x) with 'x' as a single input argument.")
    parser.add_argument("x", type=float, nargs='+', help="a start element x0 in a sequence.")

    parser.add_argument('--step', type=float, default=0.001, help='a gradient descent step with the valid range of [0, 1].')
    parser.add_argument('--eps1', type=float, default=0.001, help='an argumental accuracy parameter.')
    parser.add_argument('--eps2', type=float, default=0.01, help='a functional accuracy parameter.')
    parser.add_argument('--vareps', type=float, default=1e-8, help='a gradient component smoothing term. Prevents gradient component G to be 0.')
    parser.add_argument('--gamma', type=float, default=0.9, help='a momentum term [0, 1].')
    parser.add_argument('--beta1', type=float, default=0.9, help='a weight parameter for the expected value.')
    parser.add_argument('--beta2', type=float, default=0.999, help='a weight parameter for the unbiased variance.')
    parser.add_argument('-hg', type=float, default=0.0001, help='a step of the derivative partitioning grid with the range of [0, 1].')
    parser.add_argument('-k', type=int, default=0, help='a order of smoothing k. The higher value, the higher gradient precision. If equals 0, no smoothing is applied.')
    parser.add_argument('--criteria', type=str, default="grad", choices=["args", "func", "mixed", "grad"], help='an early stopping criteria.')
    parser.add_argument('--alg', type=str, default="sgd", choices=["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam"], help='')

    args = parser.parse_args()

    F = eval("lambda x: " + args.function)
    x0 = np.array(args.x)

    if args.alg == "sgd":
        x_opt, x_hist, epochs = SGD(F, x0, args.epoch, step=args.step, eps1=args.eps1, eps2=args.eps2, h=args.hg, k=args.k, criteria=args.criteria)
    elif args.alg == "momentum":
        x_opt, x_hist, epochs = Momentum(F, x0, args.epoch, args.gamma, step=args.step, eps1=args.eps1, eps2=args.eps2, h=args.hg, k=args.k, criteria=args.criteria)
    elif args.alg == "nag":
        x_opt, x_hist, epochs = NAG(F, x0, args.epoch, args.gamma, step=args.step, eps1=args.eps1, eps2=args.eps2, h=args.hg, k=args.k, criteria=args.criteria)
    elif args.alg == "adagrad":
        x_opt, x_hist, epochs = AdaGrad(F, x0, args.epoch, step=args.step, eps1=args.eps1, eps2=args.eps2, vareps=args.vareps, h=args.hg, k=args.k, criteria=args.criteria)
    elif args.alg == "rmsprop":
        x_opt, x_hist, epochs = RMSProp(F, x0, args.epoch, step=args.step, eps1=args.eps1, eps2=args.eps2, vareps=args.vareps, beta=args.beta1, h=args.hg, k=args.k, criteria=args.criteria)
    elif args.alg == "adam":
        x_opt, x_hist, epochs = Adam(F, x0, args.epoch, step=args.step, eps1=args.eps1, eps2=args.eps2, vareps=args.vareps, beta1=args.beta1, beta2=args.beta2, h=args.hg, k=args.k, criteria=args.criteria)

    print(f"Optima element: {x_opt}")
    print(f"Optima function value: {F(x_opt)}")
    print(f"Total iterations spent: {epochs}")

if __name__ == "__main__":
    main()
