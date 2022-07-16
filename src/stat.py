import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alg import *


### INPUT PARAMS
F = lambda x: np.sum([(1 - x[i - 1]) ** 2 + 100 * (x[i] - x[i - 1] ** 2) ** 2 for i in range(1, len(x))])
x0 = np.array([-1.5, -1.5, -1.5, -1.5, -1.5])
criteria: CriteriaType = "grad"
top_iter = 1000000
step = 0.0001
gamma = 0.9
ada_step = 0.1
eps1 = 0.01
ada_eps1 = 0.1
rmsprop_beta = 0.999
###

df = pd.DataFrame({
    "k":[], "alg":[], "x_opt":[], "epochs":[]
})

for k in [0, 3, 5, 7, 10, 20, 50]:
    sgd_x_opt, _, sgd_epochs = SGD(F, x0, top_iter, eps1=eps1, step=step, k=k, criteria=criteria)
    momentum_x_opt, _, momentum_epochs = Momentum(F, x0, top_iter, gamma, eps1=eps1, step=step, k=k, criteria=criteria)
    nag_x_opt, _, nag_epochs = NAG(F, x0, top_iter, gamma, eps1=eps1, step=step, k=k, criteria=criteria)
    adagrad_x_opt, _, adagrad_epochs = AdaGrad(F, x0, top_iter, eps1=ada_eps1, step=ada_step, k=k, criteria=criteria)
    rmsprop_x_opt, _, rmsprop_epochs = RMSProp(F, x0, top_iter, eps1=ada_eps1, step=step, beta=rmsprop_beta, k=k, criteria=criteria)
    adam_x_opt, _, adam_epochs = Adam(F, x0, top_iter, eps1=ada_eps1, step=step, k=k, criteria=criteria)

    df_row = pd.DataFrame({
        "k":[k, k, k, k, k, k],
        "alg":["sgd", "momentum", "nag", "adagrad", "rmsprop", "adam"],
        "x_opt":[sgd_x_opt, momentum_x_opt, nag_x_opt, adagrad_x_opt, rmsprop_x_opt, adam_x_opt],
        "f_opt": [F(sgd_x_opt), F(momentum_x_opt), F(nag_x_opt), F(adagrad_x_opt), F(rmsprop_x_opt), F(adam_x_opt)],
        "epochs":[sgd_epochs, momentum_epochs, nag_epochs, adagrad_epochs, rmsprop_epochs, adam_epochs]
    })
    df = pd.concat([df, df_row], ignore_index=True)

print(df.pivot(index='k', columns='alg', values=['f_opt', 'epochs']).round(6))
df.pivot(index='k', columns='alg', values='epochs').plot.bar(cmap="coolwarm")

plt.show()
