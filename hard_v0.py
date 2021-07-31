import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

N = 10
T = 300
m = 20  # number of devices
n = 5  # number of servers
y_max = 110
y_min = 90
rs = 0.1
beta = np.ones(n)

def optimization(m, n, wS, y, mu, BS, beta, CD):
    x = cp.Variable((n + 1, m), nonneg=True)
    obj = beta @ cp.inv_pos(cp.sqrt(x[1:, :] @ y + wS))

    constraints = [0 <= x, x <= 1,
                   x[1:, :] @ y <= BS,
                   cp.sum(x, 0) == 1]
    for i in range(m):
        constraints.append((y[i] * mu[i, :] @ x[1:, i]) <= CD[i])

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()  # Returns the optimal value.
    return x, prob

def oracle(y, mu):
    x, prob = optimization(m, n, wS, y, mu, BS, beta, CD)
    return x.value, prob.value, prob.status

def f(x, y):
    return beta.dot(1/np.sqrt(x[1:, :].dot(y) + wS))

reg = np.zeros((N, T))

#records all y, x_opt, x_t #yuhang yao
y_N_T = np.zeros((N, T, m)) #yuhang yao
x_opt_N_T = np.zeros((N, T, n + 1, m)) #yuhang yao
x_t_N_T = np.zeros((N, T, n + 1, m)) #yuhang yao
j_N_T = np.zeros((N, T, m))#yuhang yao

for u in range(N):
    wS = np.random.randint(15, 25, n)
    BS = np.random.uniform(y_min*5, y_max*5, n)
    CD = np.random.uniform(y_min/2, y_max/2, m)
    mu = np.random.rand(m, n)
    # trace_gen = Trace(m, n, seed + u)
    # mu = trace_gen.avg()
    # mu = np.random.rand(m, n)
    # mu_hat = np.zeros_like(mu)  # empirical mean
    mu_hat = np.ones_like(mu)
    T_ij = np.ones_like(mu)  # total number of times arm (i,j) is played
    for t in range(T):
        y = np.random.uniform(y_min, y_max, m).astype(int)
        x_opt, f_opt, status = oracle(y, mu)
        if 'optimal' not in status:
            print('Solution infeasible 1')
            break

        rho_ij = np.sqrt(3 * np.log(t + 1) / (2 * T_ij)) * rs
        mu_bar = np.maximum(mu_hat - rho_ij, 0) # LCB
        x_tmp, f_tmp, status = oracle(y, mu_bar)
        if 'optimal' not in status:
            print('Solution infeasible 2')
            break
        
        # mapping from x_tmp to x_t (new)
        x_t = np.zeros((n + 1, m))
        for i in range(m):
            cost = y[i] * mu[i,:].dot(x_tmp[1:, i])
            if cost > CD[i]:
                x_t[1:, i] = mu_bar[i] / mu[i] * x_tmp[1:, i]
                x_t[0, i] = 1-np.sum(x_t[1:, i])
                if x_t[0, i] < 0:
                    x_t[0, i] = 0
                    x_t[1:, i] = x_t[1:, i]/np.sum(x_t[1:, i])
            else:
                x_t[:, i] = x_tmp[:, i]

        f_t = f(x_t, y)

        # sample j based on x_t[i], observe c_ij, update mu_hat[i,j]
        # c = trace_gen.generate()
        for i in range(m):
            j = np.random.choice(n+1, p=x_t[:, i])
            j_N_T[u, t, i] = j #yuhang yao
            if j != 0:
                j -= 1
                c_ij = int(np.random.rand() < mu[i, j])
                # a = np.random.rand() * 3
                # c_ij = np.random.beta(a, a * (1-mu[i, j])/mu[i, j]) # beta distribution
                # c_ij = c[i, j]  # trace
                T_ij[i, j] += 1
                mu_hat[i, j] += (c_ij - mu_hat[i, j]) / T_ij[i, j]

        # calculate regert
        reg[u, t] = f_t - f_opt
        y_N_T[u, t] = y#yuhang yao
        x_opt_N_T[u, t] = x_opt#yuhang yao
        x_t_N_T[u, t] = x_t#yuhang yao
        
plt.plot(np.cumsum(reg, axis=1).T)