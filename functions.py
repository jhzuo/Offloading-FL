import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

class Trace(object):
    def __init__(self, m, n, seed=1):
        np.random.seed(seed)
        data = np.loadtxt('trace_data/Lumos5G-v1.0/Lumos5G-v1.0.csv', delimiter=',', skiprows=1,
                          usecols=(0, 15)).astype(int)
        self.traces = []
        self.m = m
        self.n = n
        for run in np.unique(data[:, 0]):
            trace = data[data[:, 0] == run, 1]
            trace = 1 / (trace + 1)
            self.traces.append(trace)
        self.average = [np.average(trace) for trace in self.traces]
        self.trace_selection = np.random.choice(len(self.traces), (m, n))
        self.trace_counter = np.zeros_like(self.trace_selection)
        for i in range(m):
            for j in range(n):
                self.trace_counter[i, j] = np.random.choice(len(self.traces[self.trace_selection[i, j]]))

    def generate(self):
        rt = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                rt[i, j] = self.traces[self.trace_selection[i, j]][self.trace_counter[i, j]]
                self.trace_counter[i, j] += 1
                if self.trace_counter[i, j] == len(self.traces[self.trace_selection[i, j]]):
                    self.trace_counter[i, j] = 0
        return rt

    def avg(self):
        rt = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                rt[i, j] = self.average[self.trace_selection[i, j]]
        return rt

#
# def objective(x, y, mu, wD, wS, m, n, gamma, alpha, BD, BS, CD=None):
#     a = np.multiply(x[0, :], y)
#     at = np.minimum(a, BD)
#     b = x[1:, :] @ y
#     bt = np.minimum(b, BS)
#     c = np.multiply(y, np.diag(mu @ x[1:, :]))
#     ct = c
#     if CD is not None:
#         ct = np.minimum(c, CD)
#
#     d = np.sum(alpha / np.sqrt(at + wD)) \
#         + np.sum(1 / np.sqrt(np.sqrt(bt + wS)))
#     e = np.sum(ct)
#     return d + gamma * e, np.average(at), np.average(bt), np.average(ct), d, gamma * e


def objective(x, y, mu, wD, wS, m, n, gamma, alpha, beta, BD, BS, CD=None):
    a = np.multiply(x[0, :], y)
    at = np.minimum(a, BD)
    b = x[1:, :] @ y
    bt = np.minimum(b, BS)
    c = np.multiply(y, np.diag(mu @ x[1:, :]))
    ct = c
    if CD is not None:
        ct = np.minimum(c, CD)

    d = np.sum(alpha / np.sqrt(at + wD)) \
        + np.sum(beta / np.sqrt(np.sqrt(bt + wS)))
    e = np.sum(ct)
    return d + gamma * e, np.average(at), np.average(bt), np.average(ct), d, gamma * e


def optimization(m, n, wD, wS, y, mu, BD, BS, gamma, hard, alpha, beta, CD=None):
    if hard:
        gamma = 0
        assert CD is not None, 'Hard CD constraint missing'

    Y = cp.Parameter((m,), nonneg=True)
    Y.value = y

    x = cp.Variable((n + 1, m), nonneg=True)

    # obj = cp.sum(alpha * cp.inv_pos(cp.sqrt(cp.multiply(x[0, :], y) + wD))) \
    #       + cp.sum(cp.inv_pos(cp.sqrt(x[1:, :] @ y + wS))) \
    #       + gamma * cp.sum(cp.multiply(cp.diag(mu @ x[1:, :]), y) )
    obj = cp.sum(alpha * cp.inv_pos(cp.sqrt(cp.multiply(x[0, :], y) + wD))) \
          + beta @ cp.inv_pos(cp.sqrt(x[1:, :] @ y + wS)) \
          + gamma * cp.sum(cp.multiply(cp.diag(mu @ x[1:, :]), y) )
    constraints = [0 <= x, x <= 1,
                   cp.multiply(x[0, :], y) <= BD,
                   x[1:, :] @ y <= BS,
                   cp.sum(x, 0) == 1]

    if hard:
        for i in range(m):
            constraints.append(y[i] * cp.sum(cp.multiply(mu[i, :], x[1:, i])) <= CD[i])

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()  # Returns the optimal value.
    return x, prob