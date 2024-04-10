import numpy as np
import cvxpy as cp

N = 25
k = 25
p = 0.2
np.random.seed(1)
A = np.random.binomial(1, p=p, size=(k, N))
print("set system:")
print(A)

x = cp.Variable(k, boolean=True)
problem = cp.Problem(cp.Minimize(cp.sum(x)), [x @ A >= 1])
problem.solve(verbose=True)
x = x.value
print("solution vector:")
print(x.astype(int))
print("minimal set cover:")
print(A[x.astype(bool)])
