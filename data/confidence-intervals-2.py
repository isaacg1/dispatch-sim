panic()
# S = Bimodal, Pareto
# g = None, 1.0, 2.0, 4.0
# rho = 0.8, 0.98
# policy = LWL, Random, JSQ, RR, JSQ-2, SITA
data = [[[[[] for _ in range(7)] for _ in range(2)] for _ in range(4)]]
def add_data(filename):
    f = open(filename, "r")
    c = 0
    for line in f:
        line = line.strip()
        if line[0] == '0':
            rho = c % 2
            g = (c//2) % 4
            s = (c//8) % 2
            values = [float(d) for d in line.split(',')[1:]]
            for i, value in enumerate(values):
                data[0][g][rho][i].append(value)
            c += 1
add_data("pareto-revised.txt")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
names="LWL,Random,JSQ,RR,JSQ-2,SITA-E,FPI".split(",")
order = [5, 0, 1, 3, 4, 2, 6]
ordered_names = [names[o] for o in order]
maxes = [[85, 200]]
bar_width = 0.16
g_names = ["g=1", "g=2", "g=4","No guardrails"]
g_colors = ["orange", "green", "purple", "0.5"]
rho_names = ["80", "98"]
plot_names = ["bp2"]
for i_s, s in enumerate(data):
    plot_name = plot_names[i_s]
    ordered_s = [s[1], s[2], s[3], s[0]]
    for i_rho in range(2):
        rho_name = rho_names[i_rho]
        plt.figure(figsize=(6, 4.5))
        for i_g, g in enumerate(ordered_s):
            rho = g[i_rho]
            means = []
            width95 = []
            for pol in rho:
                mean = sum(pol)/len(pol)
                var = sum((p-mean)**2 for p in pol)/len(pol)
                stddev = var**(0.5)
                stderr = stddev / (len(pol))**0.5
                means.append(mean)
                width95.append(1.96*stderr)
                #print(1.96*stderr/mean * 100, mean, 1.96*stderr, len(pol))
            ordered_means = [means[o] for o in order]
            ticks = np.arange(len(order))
            ax = plt.bar(ticks + bar_width * i_g, ordered_means, bar_width,
		yerr=stderr, ecolor='black',
		color = g_colors[i_g], label = g_names[i_g])
            rects = ax.patches
            for rect_index, rect in enumerate(rects):
                height = rect.get_height()
                if height > maxes[i_s][i_rho]:
                    plt.text(rect.get_x() + rect.get_width() * 1.2,
                            maxes[i_s][i_rho] * 0.93,
                            "{:.1f}".format(ordered_means[rect_index]))
            if i_g % 3 == 0:
                print(ordered_means[2:4], "{}, rho={}, s={}".format(g_names[i_g], rho_names[i_rho], plot_names[i_s]))
        plt.ylim(ymax=maxes[i_s][i_rho])
        plt.legend()
        plt.ylabel("Mean response time (E[T])")
        plt.xlabel("Dispatching policy")
        plt.xticks(ticks + len(s)/2 * bar_width, ordered_names)

        plt.tight_layout()
        plt.savefig('{}-many-pol-g-{}.eps'.format(plot_names[i_s], rho_names[i_rho]))
        plt.close()

