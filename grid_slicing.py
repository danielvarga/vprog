import numpy as np
import cvxpy as cp
from fractions import Fraction
import matplotlib.pyplot as plt
import sys


# returns an a/b Fraction. it answers this question:
# at what rational (a/b, 0) does the line between p1 and p2 intersect the x axis?
# not vectorized.
def intersect(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dy = y2 - y1
    dx = x2 - x1
    a = x1 * dy - y1 * dx
    if dy == 0:
        return Fraction(0) # yeah i know nan != 0
    else:
        return Fraction(a, dy)


def test_intersect():
    n = 20
    import matplotlib.pyplot as plt
    for _ in range(5):
        ps = np.random.randint(n, size=(3, 2))
        x = float(intersect(ps[0], ps[1]))
        # x = pq[0] / pq[1]
        ps = ps.astype(float)
        ps[2, 0] = x
        ps[2, 1] = 0
        plt.scatter(ps[:, 0], ps[:, 1])
    plt.show()

# test_intersect() ; exit()


def make_grid(n):
    l = np.arange(n, dtype=int)
    g = np.stack(np.meshgrid(l, l), axis=-1)
    gf = g.reshape(-1, 2)
    return gf


# when you connect any pair of grid points with a line,
# what are the points where these lines intersect the x axis?
def find_crossings(n):
    gf = make_grid(n)
    crossings = set()
    for p1 in gf:
        for p2 in gf:
            x = intersect(p1, p2)
            crossings.add((x.numerator, x.denominator))

    crossings = sorted([Fraction(num, den) for num, den in crossings])
    return crossings


# after crossings slice up the x axis into intervals,
# we take an internal point from each interval.
def find_midpoints(n):
    crossings = find_crossings(n)
    crossings = [crossings[0] - 1] + crossings + [crossings[-1] + 1]
    mids = []
    for i in range(len(crossings) - 1):
        mid = (crossings[i] + crossings[i + 1]) / 2
        mids.append(mid)
    return mids


def collect_star(cx, cy, n):
    gf = make_grid(n)
    gf = gf.astype(object)
    gf[:, 0] -= cx
    gf[:, 1] -= cy
    zerodivs = gf[:, 0] == 0
    assert zerodivs.sum() == 0
    derivatives = gf[:, 1] / gf[:, 0]
    derivatives = sorted(derivatives)
    mids = []
    # we leave out the vertical line,
    # will have to put them back at the very end.
    for i in range(len(derivatives) - 1):
        mid = (derivatives[i] + derivatives[i + 1]) / 2
        mids.append(mid)

    # our lines are y = mid * (x-cx) + cy
    lines = [(mid, cy - mid * cx) for mid in mids]
    return lines


# returns an (n - 1) x (n - 1) shaped grid of 0-1 pixels,
# n is the number of points, like in Go,
# not the number of pixels, like in Chess.
def digital_line(line, n):
    # ax+by+c=0 is our intersecting line
    a, b, c = line
    # evaluate the left hand side on the n x n (Go) grid:
    ev = a * np.arange(n)[:, np.newaxis] + b * np.arange(n)[np.newaxis, :] + c
    evs = np.stack([ev[:-1, :-1], ev[1:, :-1], ev[:-1, 1:], ev[1:, 1:]], axis=-1)
    assert evs.shape == (n, n, 4) # this is now a Chess grid.
    lows = np.min(evs, axis=2) < 0 # did any of its 4 corners go below 0?
    highs = np.max(evs, axis=2) > 0 # did any of its 4 corners go above 0?
    result = np.logical_and(lows, highs) # both?
    return result


digital_line((1, 1, 0.2), 10)


def vis_frac_line(line, **kwargs):
    p, q = line # y = p * x + q
    # For non-vertical lines, choose two x values
    x = np.array([-100, 100])
    # Calculate corresponding y values
    y = (p * x + q).astype(float)
    plt.plot(x, y, **kwargs)


def vis_abc_line(line, **kwargs):
    a, b, c = line
    if b != 0:
        # For non-vertical lines, choose two x values
        x = np.array([-100, 100])
        # Calculate corresponding y values
        y = (-a / b) * x - (c / b)
    else:
        # For a vertical line, x is constant
        x = np.array([-c / a, -c / a])
        # Choose y range (for example, from -10 to 10)
        y = np.array([-100, 100])
    plt.plot(x, y, **kwargs)


# for n in range(2, 25): print(f"{n}\t{len(find_midpoints(n))}")


def frac_to_abc(l):
    p, q = l # y = p*x + q line, p and q are Fractions.
    # we multiply by both denominators and reorder to 0 = ax+by+c :
    return (p.numerator * q.denominator, - p.denominator * q.denominator, p.denominator * q.numerator)


def star_test():

    n = 5

    frac_lines = collect_star(Fraction(3, 2), Fraction(10, 3), n)
    abc_lines = [frac_to_abc(l) for l in frac_lines]

    for i in range(n):
        vis_abc_line((-1, 0, i), c='lightblue')
        vis_abc_line((0, -1, i), c='lightblue')


    # for frac_line in frac_lines: vis_frac_line(frac_line)
    for abc_line in abc_lines: vis_abc_line(abc_line)

    plt.xlim(-0.5, 2*n-0.5)
    plt.ylim(-0.5, 2*n-0.5)
    plt.show()
    exit()


# star_test() ; exit()


n, = map(int, sys.argv[1:])


mids = find_midpoints(n)

collection = []

for mid in mids:
    # print(mid)
    frac_lines = collect_star(mid, Fraction(0), n)
    # for frac_line in frac_lines: print(frac_line)

    abc_lines = [frac_to_abc(l) for l in frac_lines]
    for abc_line in abc_lines:
        dl = digital_line(abc_line, n).astype(int)
        # print(dl)
        collection.append(dl)


# does not do unique
def symmetries(c):
    assert c.shape[1] == c.shape[2]
    m = c.shape[1]
    cs = []
    for cnt in range(4):
        c_prime = np.rot90(c, k=cnt, axes=(1, 2))
        cs.append(c_prime)
    cs = np.array(cs)
    cs = cs.reshape(-1, m, m)
    return cs


collection = np.array(collection)
collection = symmetries(collection)

collection = collection.reshape(len(collection), -1)
collection = np.unique(collection, axis=0)
collection = collection.reshape(len(collection), n-1, n-1)

print(len(collection))
exit()
for dl in collection:
    print(dl)

exit()


x = cp.Variable(k, boolean=True)
problem = cp.Problem(cp.Minimize(cp.sum(x)), [x @ A >= 1])
problem.solve(verbose=False)
x = x.value
print("solution vector:")
print(x.astype(int))
print("minimal set cover:")
print(A[x.astype(bool)])

