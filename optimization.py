import cvxpy as cp
import numpy as np


n = 10 # num of devices
m = 5  # num of servers


gamma = cp.Parameter((m,), nonneg=True)
gamma.value = np.ones(m)

y = cp.Parameter((n,), nonneg=True)
y.value = np.ones(n) * 100
c = np.random.random((m, n))
B = 100

x = cp.Variable((m, n))

print(cp.sqrt(x @ y).curvature)

obj = cp.Minimize(cp.sum(1 / cp.sqrt(x @ y)) )

constraints = [cp.sum(x, 0) == 1,
               0 <= x, x <= 1,
               cp.sum(c.T @ x @ y) <= B]

prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)
