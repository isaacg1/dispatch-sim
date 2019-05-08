# S = Bimodal, Pareto
# g = None, 1.0
# rho = 0.8, 0.98
# policy = LWL, Random, JSQ, RR, JSQ-2
data = [[[[[] for _ in range(7)] for _ in range(2)] for _ in range(2)] for _ in range(2)]
def add_data(filename):
    f = open(filename, "r")
    c = 0
    for line in f:
        line = line.strip()
        if line[0] == '0':
            rho = c % 2
            g = (c//2) % 2
            s = (c//4) % 2
            values = [float(d) for d in line.split(',')[1:]]
            for i, value in enumerate(values):
                if i == 5:
                    i = 6
                if len(values) == 1:
                    i = 5
                data[s][g][rho][i].append(value)
            c += 1
add_data("servers-100.txt")
add_data("server-100-sita.txt")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
names="LWL,Random,JSQ,RR,JSQ-2,SITA,FPI".split(",")
maybe_order = [5, 0, 1, 3, 4, 2, 6]
maxes = [[400, 600], [900, 1200]]
bar_width = 0.2
g_names = ["g=1","No guardrails"]
g_colors = ["orange", "0.5"]
rho_names = ["80", "98"]
plot_names = ["bm-100", "bp-100"]
for i_s, s in enumerate(data):
    plot_name = plot_names[i_s]
    ordered_s = [s[1], s[0]]
    order = maybe_order if i_s == 1 else maybe_order[:-1]
    ordered_names = [names[o] for o in order]
    for i_rho in range(2):
        rho_name = rho_names[i_rho]
        plt.figure(figsize=(6, 4.5))
        for i_g, g in enumerate(ordered_s):
            rho = g[i_rho]

            means = []
            width95 = []
            for pol in rho:
                if pol:
                    mean = sum(pol)/len(pol)
                    var = sum((p-mean)**2 for p in pol)/len(pol)
                    stddev = var**(0.5)
                    stderr = stddev / (len(pol))**0.5
                    means.append(mean)
                    width95.append(1.96*stderr)
                    #print(1.96*stderr/mean * 100, mean, 1.96*stderr, len(pol))
            ordered_means = [means[o] for o in order]
            ordered_errs = [width95[o] for o in order]
            ticks = np.arange(len(order))
            ax = plt.bar(ticks + bar_width * i_g, ordered_means, bar_width,
		yerr=ordered_errs, ecolor='black',
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

