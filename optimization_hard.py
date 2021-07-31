import numpy as np

from functions import *

seed = 1
N = 1  # number of experiments
T = 500  # rounds per experiment
rs = 0.1


np.random.seed(seed)

m = 20  # number of devices
n = 10  # number of servers
y_max = 121
y_min = 120

wD = np.ones(m) * 10
wS = np.ones(n) * 10

# wD = np.random.randint(1, 4, m)
# wS = np.random.randint(1, 4, n)

alpha = 1
beta = np.ones(n)
beta[np.random.choice(n)] = 10


# BD = np.random.uniform(y_max/n/2, y_max/n/2+5, m)
BD = np.zeros(m)
# BS = np.random.uniform(y_max * m / n - 100, y_max * m / n + 300, n)
BS = np.random.uniform(y_max * m / n +30000, y_max * m / n + 40000, n)

CD = np.random.uniform(y_min/6, y_max/6, m)
# CD = np.ones(m) * y_max

# f(x, y, mu)
def f(x, y, mu, gamma=1):
    return objective(x, y, mu, wD, wS, m, n, gamma, alpha, beta, BD, BS, CD=CD)

# min f(x, y. mu)
def oracle(y, mu, gamma=0, hard=False):
    x, prob = optimization(m, n, wD, wS, y, mu, BD, BS, gamma, hard, alpha, beta, CD=CD)
    return x.value, prob.value, prob.status


reg = np.zeros((N, T))

# statistics
stats = np.zeros((N, T, 6))

#records all y, x_opt, x_t #yuhang yao
y_N_T = np.zeros((N, T, m)) #yuhang yao
x_opt_N_T = np.zeros((N, T, n + 1, m)) #yuhang yao
x_t_N_T = np.zeros((N, T, n + 1, m)) #yuhang yao
j_N_T = np.zeros((N, T, m))#yuhang yao

for u in range(N):
    mu = np.random.rand(m, n)/2 + 0.5
    fast_link = np.random.choice(n)
    mu[:, fast_link] *= 0.02

    # mu = np.random.rand(m, n)
    # mu_hat = np.ones_like(mu)
    mu_hat = np.zeros_like(mu)  # empirical mean
    T_ij = np.ones_like(mu)  # total number of times arm (i,j) is played
    for t in range(T):
        y = np.random.uniform(y_min, y_max, m).astype(int)
        x_opt, f_opt, status = oracle(y, mu, hard=True)

        if 'optimal' not in status:
            print('Solution infeasible 1')
            break

        fv, *_, dv, ev = f(x_opt, y, mu)
        stats[u, t, :2] = np.array([dv, ev])

        rho_ij = np.sqrt(3 * np.log(t + 1) / (2 * T_ij)) * rs
        mu_bar = np.clip(mu_hat - rho_ij, 0, None) # LCB

        stats[u, t, 2] = np.average(mu_hat)
        stats[u, t, 3] = np.average(mu_bar)
        stats[u, t, 4] = np.average(np.absolute(mu - mu_bar))

        x_tmp, f_t, status = oracle(y, mu_bar, hard=True)
        if 'optimal' not in status:
            print('Solution infeasible 2')
            break
        
        # mapping from x_tmp to x_t (new)
        x_t = np.zeros((n + 1, m))
        count = 0
        for i in range(m):
            cost = y[i] * mu[i].dot(x_tmp[1:, i])
            if cost > CD[i]:
                x_t[1:, i] = mu_bar[i] / mu[i] * x_tmp[1:, i]
                x_t[0, i] = 1-np.sum(x_t[1:, i])
                if x_t[0, i] == 1:
                    x_t[1:, i] = np.ones(n) * CD[i]/y[i]/np.max(mu[i])/n
                    x_t[0, i] = 1 - np.sum(x_t[1:, i])
                    count += 1
                else:
                    x_t[1:, i] /= np.sum(x_t[1:, i])
                    x_t[0, i] = 0
            else:
                x_t[:, i] = x_tmp[:, i]

        stats[u, t, 5] = count

        # sample j based on x_t[i], observe c_ij, update mu_hat[i,j]
        x_r = np.zeros_like(x_t)
        for i in range(m):
            j = np.random.choice(n + 1, p=x_t[:, i])
            x_r[j, i] = 1
            j_N_T[u, t, i] = j #yuhang yao
            if j != 0:
                j -= 1
                # c_ij = int(np.random.rand() < mu[i, j])
                a = np.random.rand() * 3
                c_ij = np.random.beta(a, a * (1-mu[i, j])/mu[i, j]) # beta distribution
                # c_ij = c[i, j]  # trace

                T_ij[i, j] += 1
                mu_hat[i, j] += (c_ij - mu_hat[i, j]) / T_ij[i, j]

        # calculate regert
        f_t, *_ = f(x_r, y, mu, 0)
        reg[u, t] = f_t - f_opt
        
        y_N_T[u, t] = y#yuhang yao
        x_opt_N_T[u, t] = x_opt#yuhang yao
        x_t_N_T[u, t] = x_t#yuhang yao
        

plt.plot(np.cumsum(reg, axis=1).T)
obj_avg = np.average(stats[:, :, 0], 1)
cons_avg = np.average(stats[:, :, 1], 1)
legend = []
for i in range(N):
    legend.append('obj=%.2f, cst=%.2f' % (obj_avg[i], cons_avg[i]))
plt.legend(legend)
plt.title('Hard, rs=%.2f, mu_avg=%.2f' %(rs, np.average(mu)))
plt.show()

plt.plot(reg.T)
plt.title('Loss')
plt.show()

plt.plot(stats[:, :, 3].T)
plt.title('mu_bar')
plt.show()

plt.plot(stats[:, :, 4].T)
plt.title('mu-mu_bar')
plt.show()

plt.plot(stats[:, :, 5].T)
plt.title('reset counts')
plt.show()



# #yuhang yao
# Data_num_D_N_T = np.zeros((N, T, wD.shape[0]))
# Data_num_S_N_T = np.zeros((N, T, wS.shape[0]))
# for u in range(N):
#     for t in range(T):
#         Data_num_S_N_T[u, t] = wS
#         Data_num_D_N_T[u, t] = wD
#         for i in range(m):
#                     j = np.random.choice(n + 1, p=x_t_N_T[u, t, :, i])
#                     if j != 0:
#                         j -= 1
#                         Data_num_S_N_T[u, t, j] += y_N_T[u, t, i] #upload to device j-1
#                     else:
#                         Data_num_D_N_T[u, t, i] += y_N_T[u, t, i] #stay in local device
#
#
#
