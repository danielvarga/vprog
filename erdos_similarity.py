# stipped-down version of main.py, for pedagogical purposes

import numpy as np
import cvxpy as cp


n = 5
N = 2 ** n


# ChatGPT-4
def create_binary_sequences(n):
    # Generate an n-dimensional grid of indices where each dimension has size 2
    grid = np.indices((2,)*n)
    # Reshape and transpose the grid to get the desired output shape
    bit_sequences = grid.reshape(n, -1).T
    return bit_sequences


def T_omega(omega):
    ts = [1]
    for a in omega:
        t = 2 * ts[-1] + a
        ts.append(t)
    ts = np.array(ts)
    return ts


omegas = create_binary_sequences(n - 1)


Ts = []
for omega in omegas:
    Ts.append(T_omega(omega))

Ts = np.array(Ts)


print("Ts.shape", Ts.shape)


A = cp.Variable(N, boolean=True, name="A")
constraints = []

for x in range(N):
    for T in Ts:
        # A + T Minkowski sum covers x.
        # equivalently A intersects T_prime
        T_prime = [(x - y) % N for y in T]
        constraint = sum(A[y] for y in T_prime) >= 1
        constraints.append(constraint)


ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
ip.solve(solver="GUROBI", verbose=True)

A = A.value.astype(int)
print(f"{int(ip.value)}\t" + " ".join(map(str, A)))
