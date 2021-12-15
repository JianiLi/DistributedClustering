import copy
import os
from matplotlib import rc
from utils import *
from copy import deepcopy
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

plt.style.use("bmh")
plt.rcParams['axes.facecolor'] = 'white'


def noncooperative_learn(i, numAgents, psi, w, mu_k, q, attackers, psi_a):
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            # target estimation
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])
            w[:, k] = psi[:, k]
        else:
            w[:, k] = psi_a[:, a]
            a += 1

    return w


def average_learn(i, numAgents, psi, w, mu_k, q, attackers, psi_a, Neigh):
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            # target estimation
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])

        else:
            psi[:, k] = psi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            w[:, k] = np.mean(np.array([psi[:, j] for j in Neigh[k]]), axis=0)
        else:
            w[:, k] = psi[:, k]

    return w


def loss_learn(i, numAgents, x, u, d, U, D, psi, w, mu_k, q, attackers, psi_a, Accumulated_Loss, Neigh):
    a = 0
    gamma = 1
    t_last = int(i - 10) if i > 10 else 0

    for k in range(numAgents):
        if k not in attackers:
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])

        else:
            psi[:, k] = psi_a[:, a]
            a += 1

    if i < 1:
        for k in range(numAgents):
            w[:, k] = psi[:, k]
        Weight = np.zeros((numAgents, numAgents))
        np.fill_diagonal(Weight, 1)
        return w, Accumulated_Loss, Neigh, Weight

    Weight = np.zeros((numAgents, numAgents))
    reversed_loss = np.zeros((numAgents, numAgents))
    for k in range(numAgents):
        if k not in attackers:
            for l in Neigh[k]:
                # loss = (d[k] + np.dot([x[k].x, x[k].y], u[:, k].T).item() - (
                #     np.dot([psi[:, l]], u[:, k].T)).item()) ** 2
                # Accumulated_Loss[k, l] = (1 - gamma) * Accumulated_Loss[k, l] + gamma * loss
                loss = np.average([(D[0, k, t] + np.dot([x[k].x, x[k].y], U[:, k, t].T).item() - (
                    np.dot([psi[:, l]], U[:, k, t].T)).item()) ** 2 for t in range(t_last, i)])
                Accumulated_Loss[k, l] = loss

            # normalize loss
            average_loss = np.average(Accumulated_Loss[k, Neigh[k]])
            min_loss = np.min(Accumulated_Loss[k, Neigh[k]])
            argmin_loss = np.argmin(Accumulated_Loss[k, Neigh[k]])
            max_loss = np.max(Accumulated_Loss[k, Neigh[k]])

            for l in Neigh[k]:
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k] + Accumulated_Loss[k, k] * 0.0:
                    reversed_loss[k, l] = 1. / Accumulated_Loss[k, l]
                    # reversed_loss[k, l] = 1. / np.exp(Accumulated_Loss[k, l] - average_loss)
                    # reversed_loss[k, l] = 1. / np.exp((Accumulated_Loss[k, l] - min_loss) / (max_loss - min_loss))

            sum_reversedLoss = sum(reversed_loss[k, :])

            # assign weight according to 1/loss
            for l in Neigh[k]:
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k] + Accumulated_Loss[k, k] * 0.0:
                    Weight[k, l] = reversed_loss[k, l] / sum_reversedLoss
            # print(k, Weight)
            Weight[k, k] = 1 - sum(Weight[k, :k]) - sum(Weight[k, k + 1:])

            ## assign 1 weight to the one with smallest loss
            # Weight[k,argmin_loss] = 1

            w[:, k] = np.dot(psi, Weight[k, :])
        else:
            w[:, k] = psi[:, k]

    '''
    update active neighbors 
    '''
    active_neigh = []
    for k in range(numAgents):
        active_neigh_k = []
        for l in Neigh[k]:
            if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k] + Accumulated_Loss[k, k] * 0.0:
                # if Weight[k, l] > 0.02:
                active_neigh_k.append(l)
        active_neigh.append(active_neigh_k)

    return w, Accumulated_Loss, active_neigh, Weight


def distance_learn(i, numAgents, x, u, d, psi, w, mu_k, q, attackers, psi_a, Accumulated_Dist, Neigh):
    a = 0
    gamma = 1

    for k in range(numAgents):
        if k not in attackers:
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])

        else:
            psi[:, k] = psi_a[:, a]
            a += 1

    Weight = np.zeros((numAgents, numAgents))
    reversed_dist = np.zeros((numAgents, numAgents))
    for k in range(numAgents):
        if k not in attackers:
            dist = (w[0, k] - psi[0, k]) ** 2 + (w[1, k] - psi[1, k]) ** 2
            Accumulated_Dist[k, k] = (1 - gamma) * Accumulated_Dist[k, k] + gamma * dist
            for l in Neigh[k]:
                if not l == k:
                    dist = (w[0, k] - psi[0, l]) ** 2 + (w[1, k] - psi[1, l]) ** 2
                    Accumulated_Dist[k, l] = (1 - gamma) * Accumulated_Dist[k, l] + gamma * dist
                if Accumulated_Dist[k, l] <= Accumulated_Dist[k, k]:
                    reversed_dist[k, l] = (1. / Accumulated_Dist[k, l])
            sum_reversedDist = sum(reversed_dist[k, :])
            for l in Neigh[k]:
                if Accumulated_Dist[k, l] <= Accumulated_Dist[k, k]:
                    Weight[k, l] = reversed_dist[k, l] / sum_reversedDist
            # print(k, Weight)
            w[:, k] = np.dot(psi, Weight[k, :])
        else:
            w[:, k] = psi[:, k]
    '''
    update active neighbors 
    '''
    active_neigh = []
    for k in range(numAgents):
        active_neigh_k = []
        for l in Neigh[k]:
            if Accumulated_Dist[k, l] <= Accumulated_Dist[k, k] + Accumulated_Dist[k, k] * 0.0:
                # if Weight[k, l] > 0.02:
                active_neigh_k.append(l)
        active_neigh.append(active_neigh_k)

    return w, Accumulated_Dist, active_neigh, Weight


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    # parameters
    iteration = 201

    r = 1
    box = 12
    numAgents = 100
    mu_k = 0.1

    t = 1
    random1 = 0  # random.random()
    random2 = 0  # random.random()
    random3, random4 = random.random(), random.random()
    w0 = [Point(100 + t * random1, 200 + t * random1),
          Point(200 + t * random1, 100 + t * random1),
          Point(100 + t * random2, 100 + t * random2),
          Point(200 + t * random2, 200 + t * random2)]
    print([(p.x, p.y) for p in w0])
    W0 = [w0[0]] * (numAgents // 4) + [w0[1]] * (numAgents // 4) + [w0[2]] * (numAgents // 4) + [w0[3]] * (
            numAgents // 4)

    lower = 0
    upper = 3
    sensingRange = 2

    x_no = uniform_point_set(numAgents)

    x_init = copy.deepcopy(x_no)
    x_avg = copy.deepcopy(x_no)
    x_loss = copy.deepcopy(x_no)
    x_dist = copy.deepcopy(x_no)

    attackerNum = 0

    attackers = random.sample(list(range(numAgents)), attackerNum)

    normalAgents = [k for k in range(numAgents) if k not in attackers]

    Neigh = []
    for k in range(numAgents):
        neighbor = findNeighbors(x_init, k, numAgents, sensingRange, maxNeighborSize=10)
        Neigh.append(neighbor)

    psi_a = 15 + np.random.random((2, len(attackers)))

    mu_vd = 0
    mu_vu = 0
    sigma_vd2 = 0.1 + 0.1 * np.random.random((numAgents, 1))
    sigma_vd2[random.sample(range(numAgents), 5)] = 1
    sigma_vu2 = 0.01 + 0.01 * np.random.random((numAgents, 1))
    sigma_vu2[random.sample(range(numAgents), 5)] = 0.1

    vd = np.zeros((iteration, numAgents))
    vu = np.zeros((iteration, numAgents))
    for k in range(numAgents):
        vd[:, k] = np.random.normal(mu_vd, sigma_vd2[k], iteration)
        vu[:, k] = np.random.normal(mu_vu, sigma_vu2[k], iteration)

    d = np.zeros((numAgents,))
    u = np.zeros((2, numAgents))
    q = np.zeros((2, numAgents))
    psi = np.zeros((2, numAgents))

    w_no = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_no[0, k], w_no[1, k] = np.random.random(), np.random.random()
    w_avg = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_avg[0, k], w_avg[1, k] = w_no[0, k], w_no[1, k]
    w_loss = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_loss[0, k], w_loss[1, k] = w_no[0, k], w_no[1, k]
    w_dist = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_dist[0, k], w_dist[1, k] = w_no[0, k], w_no[1, k]

    Loss_no = np.zeros((iteration, numAgents))
    Loss_avg = np.zeros((iteration, numAgents))
    Loss_loss = np.zeros((iteration, numAgents))
    Loss_dist = np.zeros((iteration, numAgents))

    W1_no = np.zeros((iteration, numAgents))
    W1_avg = np.zeros((iteration, numAgents))
    W1_loss = np.zeros((iteration, numAgents))
    W1_dist = np.zeros((iteration, numAgents))

    Accumulated_Loss = np.zeros((numAgents, numAgents))
    Accumulated_Dist = np.zeros((numAgents, numAgents))

    D = np.zeros((1, numAgents, iteration))
    U = np.zeros((2, numAgents, iteration))

    Weight_loss = np.zeros((numAgents, numAgents, iteration))
    Weight_dist = np.zeros((numAgents, numAgents, iteration))

    for i in range(iteration):
        for k in range(numAgents):
            if k in attackers:
                continue
            dist = W0[k].distance(x_init[k])
            unit = [(W0[k].x - x_init[k].x) / dist, (W0[k].y - x_init[k].y) / dist]
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([W0[k].x - x_init[k].x, W0[k].y - x_init[k].y], u[:, k].T) + vd[i, k]
            q[:, k] = [x_init[k].x, x_init[k].y] + d[k] * u[:, k]
            D[:, k, i] = d[k]
            U[:, k, i] = u[:, k]

        # noncooperative
        w_no = noncooperative_learn(i, numAgents, psi, w_no, mu_k, q, attackers, psi_a)

        # cooperative
        w_avg = average_learn(i, numAgents, psi, w_avg, mu_k, q, attackers, psi_a, Neigh)

        w_loss, Accumulated_Loss, active_neigh_loss, weight_loss = loss_learn(i, numAgents, x_loss, u, d, U, D, psi,
                                                                              w_loss, mu_k, q,
                                                                              attackers,
                                                                              psi_a,
                                                                              Accumulated_Loss, Neigh)
        Weight_loss[:, :, i] = weight_loss

        w_dist, Accumulated_Dist, active_neigh_dist, weight_dist = distance_learn(i, numAgents, x_dist, u, d, psi,
                                                                                  w_dist, mu_k,
                                                                                  q, attackers, psi_a,
                                                                                  Accumulated_Dist, Neigh)
        Weight_dist[:, :, i] = weight_dist

        # loss_no = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_no[0, k], w_no[1, k])
            # error_no += agent.distance(W0[k]) ** 2
            loss_no = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_no[:, k]], u[:, k].T)).item()) ** 2
            W1_no[i, k] = w_no[0, k]
            Loss_no[i, k] = loss_no

        # loss_avg = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_avg[0, k], w_avg[1, k])
            # error_avg += agent.distance(W0[k]) ** 2
            loss_avg = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_avg[:, k]], u[:, k].T)).item()) ** 2
            W1_avg[i, k] = w_avg[0, k]
            Loss_avg[i, k] = loss_avg

        # loss_loss = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_loss[0, k], w_loss[1, k])
            # error_loss += (agent.distance(W0[k])) ** 2
            loss_loss = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_loss[:, k]], u[:, k].T)).item()) ** 2
            W1_loss[i, k] = w_loss[0, k]
            Loss_loss[i, k] = loss_loss

        # if i % 100 == 0:
        if i == iteration - 1:
            fig_loss_cluster = plt.figure(figsize=(4, 4))
            for k in range(0, numAgents):
                if k in attackers:
                    continue
                for neighbor in active_neigh_loss[k]:
                    plt.plot([x_init[k].x, x_init[neighbor].x], [x_init[k].y, x_init[neighbor].y], linewidth=0.8,
                             color='k')
            plot_point_set(x_init[:numAgents // 4], color='#CCFFFF')  # fault-free robots are plotted in blue
            plot_point_set(x_init[numAgents // 4:numAgents // 2],
                           color='#CCFF99')  # fault-free robots are plotted in blue
            plot_point_set(x_init[numAgents // 2:numAgents // 4 * 3],
                           color='#FFFF99')  # fault-free robots are plotted in GREEN
            plot_point_set(x_init[numAgents // 4 * 3:], color='#FFCCCC')  # fault-free robots are plotted in GREEN
            plot_point_set([x_init[p] for p in attackers], color='#FFE6E6')  # faulty robots are plotted in red
            plt.show()

            fig_dist_cluster = plt.figure(figsize=(4, 4))
            for k in range(0, numAgents):
                if k in attackers:
                    continue
                for neighbor in active_neigh_dist[k]:
                    plt.plot([x_init[k].x, x_init[neighbor].x], [x_init[k].y, x_init[neighbor].y], linewidth=0.8,
                             color='k')
            plot_point_set(x_init[:numAgents // 4], color='#CCFFFF')  # fault-free robots are plotted in blue
            plot_point_set(x_init[numAgents // 4:numAgents // 2],
                           color='#CCFF99')  # fault-free robots are plotted in blue
            plot_point_set(x_init[numAgents // 2:numAgents // 4 * 3],
                           color='#FFFF99')  # fault-free robots are plotted in GREEN
            plot_point_set(x_init[numAgents // 4 * 3:], color='#FFCCCC')  # fault-free robots are plotted in GREEN
            plot_point_set([x_init[p] for p in attackers], color='#FFE6E6')  # faulty robots are plotted in red
            plt.show()

        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_dist[0, k], w_dist[1, k])
            # error_loss += (agent.distance(W0[k])) ** 2
            loss_dist = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_dist[:, k]], u[:, k].T)).item()) ** 2
            W1_dist[i, k] = w_dist[0, k]
            Loss_dist[i, k] = loss_dist

        print('iteration %d' % i)

    np.save('results/Weight_loss.npy', Weight_loss)
    np.save('results/Weight_dist.npy', Weight_dist)

    Loss_mean, Loss_min, Loss_max, Loss_std = {}, {}, {}, {}
    for rule in ["no", "avg", "dist", "loss"]:
        Loss_mean[rule] = np.mean(np.delete(locals()["Loss_" + rule], attackers, axis=1), 1)
        Loss_min[rule] = np.min(np.delete(locals()["Loss_" + rule], attackers, axis=1), 1)
        Loss_max[rule] = np.max(np.delete(locals()["Loss_" + rule], attackers, axis=1), 1)
        Loss_std[rule] = np.std(np.delete(locals()["Loss_" + rule], attackers, axis=1), 1)

        np.save('results/Loss_mean.npy', Loss_mean)
        np.save('results/Loss_min.npy', Loss_min)
        np.save('results/Loss_max.npy', Loss_max)
        np.save('results/Loss_std.npy', Loss_std)

    fig = plt.figure(figsize=(7, 4))
    legend = {"no": "no-coop", "avg": "average", "loss": "loss-based", "dist": "distance-based"}
    for rule in ["no", "avg", "dist", "loss"]:
        try:
            plt.plot(Loss_mean[rule], label=legend[rule])
            plt.fill_between(range(len(Loss_min[rule])), Loss_min[rule], Loss_max[rule], alpha=0.3)
            plt.yscale('log')
        except:
            pass

    plt.xlabel(r'iteration $i$', fontsize=20)
    plt.ylabel(r'Loss', fontsize=20)
    plt.legend(fontsize=15, loc='center left', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    plt.show()

    fig3 = plt.figure(figsize=(11, 2.5))
    plt.subplot(141)
    for k in normalAgents:
        plt.plot(W1_no[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$\theta_{k,i}(1)$', fontsize=25)
    fig3.tight_layout(pad=1.0)
    plt.subplot(142)
    for k in normalAgents:
        plt.plot(W1_avg[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$\theta_{k,i}(1)$', fontsize=25)
    fig3.tight_layout(pad=1.0)
    plt.subplot(143)
    for k in normalAgents:
        plt.plot(W1_loss[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$\theta_{k,i}(1)$', fontsize=25)
    fig3.tight_layout(pad=1.0)
    plt.subplot(144)
    for k in normalAgents:
        plt.plot(W1_dist[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$\theta_{k,i}(1)$', fontsize=25)

    plt.show()
