import random
import math
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point


def random_point_set(n, lower=-10, upper=10):
    points = []
    assert lower <= upper
    for i in range(n):
        x = random.uniform(lower, upper)
        y = random.uniform(lower, upper)
        points.append(Point(x, y))
    return points

def uniform_point_set(n):
    points = [Point(0.0, 0.0)]*n
    width = int(math.sqrt(n))

    indexs = list(range(0, n))
    random.shuffle(indexs)

    for i in range(width):
        for j in range(width):
            x = random.uniform(i, i + 1)
            y = random.uniform(j, j + 1)
            points[indexs[i*width + j]] = Point(x, y)

    return points

def plot_point_set(point_set, color='b', ax=None, alpha=1):
    for p in point_set:
        plot_point(p, color=color, ax=ax, alpha=alpha)


def plot_point(P, marker='o', color='b', size=10, ax=None, alpha=1):
    if ax == None:
        plt.plot(P.x, P.y, marker=marker, color=color, markersize=size, markeredgecolor='k', markeredgewidth=0.5, alpha=alpha)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
    else:
        ax.plot(P.x, P.y, marker=marker, color=color, markersize=size, markeredgecolor='k', markeredgewidth=0.5, alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)



def findNeighbors(x, k, numAgents, rmax, maxNeighborSize=10):
    N = [k]
    for i in range(numAgents):
        if i == k:
            continue
        n = x[i]
        if np.sqrt((n.y - x[k].y) ** 2 + (n.x - x[k].x) ** 2) <= rmax:
            N.append(i)

    if len(N) > maxNeighborSize:
        selection = random.sample(N[1:], maxNeighborSize-1)
        return [N[0]]+selection
    else:
        return N


def h(w, x, s=1):
    dist = w.distance(x)
    if dist <= s:
        return np.array([w.x - x.x, w.y - x.y])
    else:
        return np.array([w.x - x.x, w.y - x.y]) / dist * s


def Delta(x, k, numAgents, r=2, sensingRange=10):
    N = []
    for l in range(numAgents):
        if x[l].distance(x[k]) <= sensingRange:
            N.append(l)
    delta = 0
    for l in N:
        if l == k:
            continue
        dist = x[l].distance(x[k])
        if dist != 0:
            delta += np.array([x[l].x - x[k].x, x[l].y - x[k].y]) / dist * (dist - r)

    #return delta * (1 / len(N))
    return delta