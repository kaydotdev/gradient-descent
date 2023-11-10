import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.alg import CriteriaType, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam

# INPUT PARAMS
f = lambda x: np.sum([(1 - x[i - 1]) ** 2 + 100 * (x[i] - x[i - 1] ** 2) ** 2 for i in range(1, len(x))])
x0 = np.array([-1.5, -1.5, -1.5, -1.5, -1.5])
criteria: CriteriaType = "grad"
top_iter = 10000
step = 0.001
gamma = 0.9
ada_step = 0.1
eps1 = 0.01
ada_eps1 = 0.1
beta1 = 0.9
beta2 = 0.999
###

df = pd.DataFrame({
    "k": [], "alg": [], "x_opt": [], "epochs": []
})

for k in [0, 3, 5, 7, 10, 20, 50]:
    sgd_x_opt, _, sgd_epochs = SGD(f, x0, top_iter, eps1=eps1, step=step, k=k, criteria=criteria)
    momentum_x_opt, _, momentum_epochs = Momentum(f, x0, top_iter, gamma, eps1=eps1, step=step, k=k, criteria=criteria)
    nag_x_opt, _, nag_epochs = NAG(f, x0, top_iter, gamma, eps1=eps1, step=step, k=k, criteria=criteria)
    adagrad_x_opt, _, adagrad_epochs = AdaGrad(f, x0, top_iter, eps1=ada_eps1, step=ada_step, k=k, criteria=criteria)
    rmsprop_x_opt, _, rmsprop_epochs = RMSProp(f, x0, top_iter, eps1=ada_eps1, step=step, beta=beta1, k=k,
                                               criteria=criteria)
    adam_x_opt, _, adam_epochs = Adam(f, x0, top_iter, eps1=ada_eps1, step=step, beta1=beta1, beta2=beta2, k=k,
                                      criteria=criteria)

    df_row = pd.DataFrame({
        "k": [k, k, k, k, k, k],
        "alg": ["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam"],
        "x_opt": [sgd_x_opt, momentum_x_opt, nag_x_opt, adagrad_x_opt, rmsprop_x_opt, adam_x_opt],
        "f_opt": [f(sgd_x_opt), f(momentum_x_opt), f(nag_x_opt), f(adagrad_x_opt), f(rmsprop_x_opt), f(adam_x_opt)],
        "epochs": [sgd_epochs, momentum_epochs, nag_epochs, adagrad_epochs, rmsprop_epochs, adam_epochs]
    })
    df = pd.concat([df, df_row], ignore_index=True)

df.round(6).to_csv("artifacts/benchmark.csv")
df.pivot(index='k', columns='alg', values=['f_opt', 'epochs']).round(6).to_csv("artifacts/table.csv")
df.pivot(index='k', columns='alg', values='epochs').plot.bar(cmap="coolwarm")

plt.show()
