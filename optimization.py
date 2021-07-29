import cvxpy as cp
import numpy as np


n = 10 # num of devices
m = 5  # num of servers


gamma = cp.Parameter((m,), pos=True)
gamma.value = np.ones(m)

y = cp.Parameter((n,), nonneg=True)
y.value = np.ones(n) * 100
c = np.random.random((m, n))
B = 5000


def solver(wD, wS, y, u, BD, BS, gamma = 0, hard=False):

    x = cp.Variable((m, n), nonneg=True)

    obj = cp.Minimize(cp.sum(cp.inv_pos(cp.sqrt(x @ y)) ) )

    constraints = [0 <= x, x <= 1,
               cp.sum(c.T @ x @ y) <= B,
               cp.sum(x, 0) == 1]

    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)

solver(None, None, None, None, None)
