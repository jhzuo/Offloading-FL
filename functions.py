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


def objective(x, y, mu, wD, wS, m, n, gamma, alpha):
    a = np.multiply(x[0, :], y)
    b = x[1:, :] @ y
    c = mu @ b
    d = np.sum(alpha / np.sqrt(a + wD)) \
        + np.sum(1 / np.sqrt(np.sqrt(b + wS)))
    e = np.sum(c) / m / n
    return d + gamma * e, np.average(a), np.average(b), np.average(c), np.average(d), gamma * np.average(e)


def optimization(m, n, wD, wS, y, mu, BD, BS, gamma, hard, alpha, CD=None):
    if hard:
        gamma = 0
        assert CD is not None, 'Hard CD constraint missing'

    Y = cp.Parameter((m,), nonneg=True)
    Y.value = y

    x = cp.Variable((n + 1, m), nonneg=True)

    obj = cp.sum(alpha * cp.inv_pos(cp.sqrt(cp.multiply(x[0, :], y) + wD))) \
          + cp.sum(cp.inv_pos(cp.sqrt(x[1:, :] @ y + wS))) + gamma * cp.sum(mu @ (x[1:, :] @ y)) / m / n

    constraints = [0 <= x, x <= 1,
                   cp.multiply(x[0, :], y) <= BD,
                   x[1:, :] @ y <= BS,
                   cp.sum(x, 0) == 1]

    if hard:
        constraints.append(mu @ (x[1:, :] @ y) <= CD)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()  # Returns the optimal value.
    return x, prob