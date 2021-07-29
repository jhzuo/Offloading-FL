
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


m = 20 # number of devices
n = 5  # number of servers

wD = np.random.randint(1 ,4 ,m )
wS = np.random.randint(1 ,4 ,n )

BD = 5000
BS = 5000
CD = 5000


# f(x, y, mu)
def f(x, y, mu, gamma=0):
    a = np.sum( 1 /np.sqrt(np.multiply(x[0, :], y) + wD)) \
           + np.sum( 1 /np.sqrt(np.sqrt(x[1:, :] @ y+ wS)))
    b = np.sum( (mu @ x[1:, :] @ y))
    return a + gamma * b


# min f(x, y. mu)
def oracle(y, mu, gamma=0, hard=False):
    if hard:
        gamma = 0

    Y = cp.Parameter((m,), nonneg=True)
    Y.value = y

    x = cp.Variable((n + 1, m), nonneg=True)


    obj = cp.sum(cp.inv_pos(cp.sqrt(cp.multiply(x[0, :], y) + wD))) \
          + cp.sum(cp.inv_pos(cp.sqrt(x[1:, :] @ y + wS))) + gamma * cp.sum(mu @ (x[1:, :] @ y))

    constraints = [0 <= x, x <= 1,
                   cp.multiply(x[0, :], y) <= BD,
                   x[1:, :] @ y <= BS,
                   cp.sum(x, 0) == 1]

    if hard:
        constraints.append(mu @ (x[1:, :] @ y) <= CD)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()  # Returns the optimal value.
    return x.value, prob.value, prob.status,


N = 1  # number of experiments
T = 300  # rounds per experiment
reg = np.zeros((N, T))
gamma = 1
for u in range(N):
    mu = np.random.rand(m, n)  # real mu
    mu_hat = np.zeros_like(mu)  # empirical mean
    T_ij = np.ones_like(mu)  # total number of times arm (i,j) is played
    for t in range(T):
        y = np.random.uniform(80, 120, m).astype(int)
        x_opt, f_opt, status = oracle(y, mu, gamma)

        rho_ij = np.sqrt(3 * np.log(t + 1) / (2 * T_ij))
        mu_bar = np.max(mu_hat - rho_ij, 0)  # LCB

        x_t, f_t, status= oracle(y, mu_bar, gamma)
        f_t = f(x_t, y, mu, gamma)

        # sample j based on x_t[i], observe c_ij, update mu_hat[i,j]        
        for i in range(m):
            j = np.random.choice(n + 1, p=x_t[:, i])
            if j != 0:
                j -= 1
                c_ij = int(np.random.rand() < mu[i, j])
                T_ij[i, j] += 1
                mu_hat[i, j] += (c_ij - mu_hat[i, j]) / T_ij[i, j]

        # calculate regert
        reg[u, t] = f_t - f_opt

plt.plot(reg.T)
plt.show()