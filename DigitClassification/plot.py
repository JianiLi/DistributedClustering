import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 155
plt.rcParams['axes.facecolor'] = 'white'

loss_or_acc = "loss"
attack = False
attackerNum = 0
data = "synthetic"  # "mnist"

def print_num(l, string, num=5):
    print(string + "=[", end=' ')
    for i in range(len(l)):
        print("%.7f" % l[i], end='  ')
        if (i + 1) % num == 0:
            print()
    print("]")


N = 20
N1 = 0  # N // 2

train_Loss_mean, train_Loss_min, train_Loss_max, train_Loss_std = {}, {}, {}, {}
train_Acc_mean, train_Acc_min, train_Acc_max, train_Acc_std = {}, {}, {}, {}

test_Loss_mean, test_Loss_min, test_Loss_max, test_Loss_std = {}, {}, {}, {}
test_Acc_mean, test_Acc_min, test_Acc_max, test_Acc_std = {}, {}, {}, {}

for train_test in ["train", "test"]:
    for rule in ["no-cooperation", "average", "distance", "loss"]:
        if attack:
            loss = np.load("saved_results/attacked/individual_average_%s_loss_%s.npy" % (train_test, rule))
            acc = np.load("saved_results/attacked/individual_average_%s_acc_%s.npy" % (train_test, rule))
        else:
            loss = np.load("saved_results/individual_average_%s_loss_%s.npy" % (train_test, rule))
            acc = np.load("saved_results/individual_average_%s_acc_%s.npy" % (train_test, rule))

        if data == "mnist":
            loss = loss[:, :N1]
            acc = acc[:, :N1] * 100
        else:
            loss = loss[:, N1:]
            acc = acc[:, N1:] * 100

        mean_loss = np.nanmean(loss, 1)
        max_loss = np.nanmax(loss, 1)
        min_loss = np.nanmin(loss, 1)
        std_loss = np.nanstd(loss, 1)

        locals()[train_test + "_Loss_mean"][rule], locals()[train_test + "_Loss_max"][rule], \
        locals()[train_test + "_Loss_min"][rule], locals()[train_test + "_Loss_std"][
            rule] = mean_loss, min_loss, max_loss, std_loss

        # accuracy
        mean_acc = np.nanmean(acc, 1)
        max_acc = np.nanmax(acc, 1)
        min_acc = np.nanmin(acc, 1)
        std_acc = np.nanstd(acc, 1)

        locals()[train_test + "_Acc_mean"][rule], locals()[train_test + "_Acc_max"][rule], \
        locals()[train_test + "_Acc_min"][rule], locals()[train_test + "_Acc_std"][
            rule] = mean_acc, min_acc, max_acc, std_acc

        if rule == "no-cooperation":
            rule = "no_cooperation"

        print_num(mean_loss, "mean_loss_%s" % rule, num=10)
        print_num(min_loss, "min_loss_%s" % rule, num=10)
        print_num(max_loss, "max_loss_%s" % rule, num=10)
        print_num(std_loss, "std_loss_%s" % rule, num=10)

        print_num(mean_acc, "mean_acc_%s" % rule, num=10)
        print_num(min_acc, "min_acc_%s" % rule, num=10)
        print_num(max_acc, "max_acc_%s" % rule, num=10)
        print_num(std_acc, "std_acc_%s" % rule, num=10)

keys = ["no-cooperation", "average", "distance", "loss"]
legend = {"no-cooperation": "no-coop", "average": "average", "loss": "loss-based", "distance": "distance-based"}
line_style = {"no-cooperation": "-", "average": "-", "loss": "-", "distance": "--"
              }
iteration = 300
y_loss_lim_train = 0.04
y_loss_lim_test = 0.05
y_acc_lim = 100

fig = plt.figure(figsize=(15, 3.5))
ax = fig.add_subplot(1, 4, 1)
for key in keys:
    ax.plot(train_Loss_mean[key][:iteration], label=legend[key], linestyle=line_style[key])
    ax.fill_between(range(iteration), train_Loss_min[key][:iteration], train_Loss_max[key][:iteration], alpha=0.3)
plt.xlabel(r"Iteration $i$", fontsize=20)
plt.ylabel(r"Train loss", fontsize=20)
plt.yticks([0, 0.008, 0.016, 0.024, 0.032, 0.04])
plt.ylim([0 - y_loss_lim_train * 0.05, y_loss_lim_train * 1.05])
plt.tight_layout()

ax = fig.add_subplot(1, 4, 2)
for key in keys:
    ax.plot(train_Acc_mean[key][:iteration], label=legend[key], linestyle=line_style[key])
    ax.fill_between(range(iteration), train_Acc_min[key][:iteration], train_Acc_max[key][:iteration], alpha=0.3)
plt.xlabel(r"Iteration $i$", fontsize=20)
plt.ylabel("Train accuracy (%)", fontsize=20)
# plt.legend(fontsize=15)
plt.ylim([0 - y_acc_lim * 0.05, y_acc_lim * 1.05])
plt.tight_layout()

ax = fig.add_subplot(1, 4, 3)
for key in keys:
    ax.plot(test_Loss_mean[key][:iteration], label=legend[key], linestyle=line_style[key])
    ax.fill_between(range(len(test_Loss_min[key][:iteration])), test_Loss_min[key][:iteration],
                    test_Loss_max[key][:iteration], alpha=0.3)
plt.xlabel(r"Iteration $i$", fontsize=20)
plt.ylabel(r"Test loss", fontsize=20)
# plt.legend(fontsize=15)
plt.ylim([0 - y_loss_lim_test * 0.05, y_loss_lim_test * 1.05])
plt.tight_layout()

ax = fig.add_subplot(1, 4, 4)
for key in keys:
    ax.plot(test_Acc_mean[key][:iteration], label=legend[key], linestyle=line_style[key])
    ax.fill_between(range(len(test_Acc_min[key][:iteration])), test_Acc_min[key][:iteration],
                    test_Acc_max[key][:iteration], alpha=0.3)
plt.xlabel(r"Iteration $i$", fontsize=20)
plt.ylabel(r"Test accuracy (%)", fontsize=20)
# plt.legend(fontsize=15)
plt.ylim([0 - y_acc_lim * 0.05, y_acc_lim * 1.05])
plt.tight_layout()

# plt.savefig('saved_results/cluster_DC_%s_learning_curve_%d.png' % (data, N), dpi=1000)
# plt.savefig('saved_results/cluster_DC_learning_curve_%d_whole.png' % (N), dpi=1000)
plt.show()
