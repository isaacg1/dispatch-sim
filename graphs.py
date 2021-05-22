import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

"""
time=1e6
S=Bimodal(1, 1000, 0.9995)
k=10
FCFS
SRPT
Guard
c=1+1/(1+ln(1/(1-rho)))
g=4

rho,lwl,random
"""
fcfs_lwl_data = """
0.020000,15.715139
0.100000,15.306055
0.200000,15.172792
0.300000,15.087364
0.400000,15.134047
0.500000,15.568631
0.600000,20.328113
0.700000,51.205623
0.750000,106.297987
0.800000,213.940471
0.850000,436.910489
0.900000,1033.928636
0.925000,1812.139313
0.950000,4388.483866
0.970000,7939.682000
"""
fcfs_random_data = """
0.020000,54.276934
0.040000,78.035510
0.060000,117.599048
0.080000,157.335962
0.100000,219.333548
0.120000,265.858276
0.140000,324.856588
0.160000,362.453878
0.180000,409.381586
0.200000,461.281807
0.220000,508.972235
0.240000,565.371828
0.260000,621.483079
""".strip()
srpt = """
0.020000,15.715139,16.064841
0.100000,15.306055,16.224027
0.200000,15.172792,17.096488
0.300000,15.087364,18.124830
0.400000,15.134014,19.527137
0.500000,15.513372,21.229960
0.600000,18.806942,23.847628
0.700000,37.633093,27.668816
0.750000,64.457091,30.463453
0.800000,113.243375,34.131625
0.850000,194.049947,39.721322
0.900000,287.738594,50.136458
0.925000,380.511102,58.260645
0.950000,486.443438,70.187874
0.970000,565.559590,84.589822
0.980000,632.026693,94.545752
0.990000,634.675484,107.010752
""".strip()
guard = """\
0.020000,15.715139,15.794545
0.100000,15.306055,15.797295
0.200000,15.172792,16.185073
0.300000,15.087382,16.547876
0.400000,15.130444,17.176808
0.500000,15.109423,17.855612
0.600000,15.395883,18.853786
0.700000,16.136970,19.690314
0.750000,16.805679,21.056429
0.800000,17.641636,22.472912
0.850000,18.723721,24.464028
0.900000,20.554664,27.426791
0.925000,22.286584,30.122730
0.950000,26.766583,35.285085
0.970000,32.584486,41.165873
0.980000,36.582673,45.495334
0.990000,42.200913,52.263039
""".strip()

fcfs_lwl_rhos, fcfs_lwl =\
    [[float(line.split(',')[i]) for line in fcfs_lwl_data.split()] for i in range(2)]

fcfs_random_rhos, fcfs_random =\
    [[float(line.split(',')[i]) for line in fcfs_random_data.split()] for i in range(2)]

srpt_rhos, srpt_lwl, srpt_random =\
    [[float(line.split(',')[i]) for line in srpt.split()] for i in range(3)]

guard_rhos, guard_lwl, guard_random =\
    [[float(line.split(',')[i]) for line in guard.split()] for i in range(3)]

plt.figure(figsize=(5, 3.25))
plt.plot(fcfs_random_rhos, fcfs_random, 'v-', label='Random/FCFS', linewidth=4, color='orange')
plt.plot(fcfs_lwl_rhos, fcfs_lwl, 'd-', label='LWL/FCFS', linewidth=4, color='cyan')
plt.plot(srpt_rhos, srpt_lwl, 's-', label='LWL/SRPT', linewidth=4, color='blue')
plt.plot(srpt_rhos, srpt_random, '^-', label='Random/SRPT', linewidth=4, color='red')

plt.ylim(ymax=500)

plt.legend(loc='upper center')
plt.ylabel("Mean response time ($E[T]$)")
plt.xlabel(r"System load ($\rho$)")
plt.tight_layout()
plt.savefig('srpt-vs-fcfs.eps', format='eps')

plt.close()

plt.figure(figsize=(5, 3.25))
plt.plot(srpt_rhos, srpt_lwl, 's-', label='LWL/SRPT', linewidth=4, color='blue')
plt.plot(srpt_rhos, srpt_random, '^-', label='Random/SRPT', linewidth=4, color='red')
plt.plot(guard_rhos, guard_random, '^', label='G-Random/SRPT', linewidth=4, color='red', linestyle='dashed')
plt.plot(guard_rhos, guard_lwl, 's', label='G-LWL/SRPT', linewidth=4, color='blue', linestyle='dashed')

plt.ylim(ymax=100)

plt.legend(loc='upper left')
plt.ylabel("Mean response time (E[T])")
plt.xlabel(r"System load ($\rho$)")
plt.tight_layout()
plt.savefig('guard-vs-unguard.eps', format='eps')

plt.close()

plt.plot(fcfs_random_rhos, fcfs_random, 'v-', label='Random/FCFS', linewidth=4, color='orange')
plt.plot(fcfs_lwl_rhos, fcfs_lwl, 'd-', label='LWL/FCFS', linewidth=4, color='cyan')
plt.plot(srpt_rhos, srpt_lwl, 's-', label='LWL/SRPT', linewidth=4, color='blue')
plt.plot(srpt_rhos, srpt_random, '^-', label='Random/SRPT', linewidth=4, color='red')
plt.plot(guard_rhos, guard_random, '^', label='G-Random/SRPT', linewidth=4, color='red', linestyle='dashed')
plt.plot(guard_rhos, guard_lwl, 's', label='G-LWL/SRPT', linewidth=4, color='blue', linestyle='dashed')

plt.ylim(ymax=200)

plt.ylabel("Mean response time (E[T])")
plt.xlabel(r"System load ($\rho$)")

plt.savefig('all-together-guard.eps', format='eps')
plt.savefig('all-together-guard.png', format='png')
"""
BP:
alpha=1.5
bound=1e6
g=2
time=3e5
k=10
"""
names="LWL,Random,JSQ,RR,JIQ,JSQ-2".split(",")
colors="red,blue,brown,purple,green,black".split(",")
order=[2,1,4,0,5,3]
big_guard = """
0.020000,30.356579,30.631346,30.356579,30.407110,30.356579,30.365475
0.100000,30.786054,32.241958,30.786054,31.426070,30.786054,30.948782
0.200000,30.136408,33.119716,30.138142,31.735982,30.137222,30.754346
0.300000,30.065806,34.631777,30.078279,33.068265,30.074833,31.384186
0.400000,29.952441,36.247102,30.013522,34.645067,30.004233,32.203288
0.500000,30.023119,38.091678,30.204729,36.591323,30.202293,33.432091
0.600000,30.341167,40.265619,30.682088,38.534162,30.746539,34.852339
0.700000,31.627187,43.491292,32.109498,41.530292,32.327873,37.172425
0.750000,32.797578,45.598331,33.217433,43.094581,33.668049,38.651027
0.800000,34.561664,48.286602,34.728220,45.201511,35.615254,40.544387
0.850000,37.826086,52.464277,37.284241,48.868927,39.119328,43.547072
0.900000,43.174617,58.577021,41.025528,53.781294,44.794883,47.585709
0.925000,47.278138,62.632995,43.461794,57.218198,49.057643,50.492118
0.950000,54.730036,69.552584,47.668902,63.477212,56.775190,55.439192
0.970000,62.609273,76.698713,51.988131,71.009341,64.789773,60.406760
0.980000,67.583790,82.739594,55.679617,75.928165,71.049894,64.079265
0.990000,75.269021,90.134344,59.771688,83.258005,80.007513,69.574167
""".strip().split("\n")
big_unguard = """
0.020000,30.356579,30.708915,30.356579,30.403216,30.356579,30.365475
0.100000,30.786054,32.513162,30.786054,31.387468,30.786054,30.975984
0.200000,30.136408,33.694527,30.138142,31.506609,30.137944,30.814458
0.300000,30.065764,35.909607,30.078697,32.651433,30.074218,31.539931
0.400000,29.952126,38.449569,30.013705,34.182409,30.008885,32.532695
0.500000,30.026117,41.695632,30.202380,36.303849,30.202402,33.998698
0.600000,30.381180,45.762634,30.670486,39.022317,30.745526,35.853762
0.700000,31.929045,51.952766,32.028947,43.353551,32.283980,38.715137
0.750000,33.481436,56.041757,33.112181,46.458462,33.592555,40.510038
0.800000,36.033476,61.172848,34.570300,50.316724,35.474253,42.699921
0.850000,41.662332,68.696289,36.981343,56.068260,38.841555,45.969088
0.900000,51.163617,78.846647,40.216515,64.323037,43.863959,50.210771
0.925000,59.408645,86.239325,42.504696,70.417950,47.792276,53.061358
0.950000,74.417453,97.681441,46.482252,80.726781,54.641323,57.602195
0.970000,88.925423,113.376553,50.464532,93.534230,62.880025,62.172966
0.980000,98.986495,124.488738,53.196507,103.078259,69.189879,65.896013
0.990000,111.856998,138.608691,59.981939,116.478546,80.101816,70.794203
""".strip().split("\n")
plt.figure(figsize=(6, 5))
big_rhos = [float(line.split(",")[0]) for line in big_unguard]
for i in order:
    unguard_times = [float(line.split(",")[i+1]) for line in big_unguard]
    plt.plot(big_rhos, unguard_times, label=names[i], linewidth=4, color=colors[i])
for i in order:
    guard_times = [float(line.split(",")[i+1]) for line in big_guard]
    plt.plot(big_rhos, guard_times, label="G-"+names[i], linewidth=4, color=colors[i], linestyle='dashed')
plt.ylim(ymax=60)

plt.legend(loc='upper left', ncol=2)
plt.ylabel("Mean response time (E[T])")
plt.xlabel(r"System load ($\rho$)")

plt.tight_layout()
#plt.savefig('many-policies.eps')

plt.close()

import numpy as np

for rho in [6, 9, 15]:
    rho_name = big_guard[rho][2:4]
    plt.figure(figsize=(6,4))
    unguard_rho = [float(f) for f in big_unguard[rho].split(",")][1:]
    ticks = np.arange(len(unguard_rho))
    bar_width = 0.35
    plt.bar(ticks, unguard_rho, bar_width, color="green", label="Unguarded")
    guard_rho = [float(f) for f in big_guard[rho].split(",")][1:]
    plt.bar(ticks + bar_width, guard_rho, bar_width, color="purple", label="Guarded")

    plt.legend()
    plt.ylabel("Mean response time (E[T])")
    plt.xlabel(r"System load ($\rho$)")
    plt.xticks(ticks + bar_width, names)

    plt.tight_layout()
#    plt.savefig('many_policies_rho_{}.eps'.format(rho_name))
    plt.close()


plt.figure(figsize=(6, 5))
new_order = [0, 1, 2, 3, 4, 5]
sample_rhos=[6,9,11,13, 15]
ticks=np.arange(len(sample_rhos))
new_rhos = [big_rhos[i] for i in sample_rhos]
bar_width = 0.15
for (index, order_i) in enumerate(new_order):
    unguard_times = [float(line.split(",")[order_i+1]) for line in big_unguard]
    guard_times = [float(line.split(",")[order_i+1]) for line in big_guard]
    ratios = [100*(guard_times[j]/unguard_times[j]-1) for j in sample_rhos]
    plt.bar(ticks + index * bar_width, ratios, bar_width, color=colors[order_i],
            label=names[order_i])
    
plt.legend(loc='lower left', bbox_to_anchor=(-0.01, 0), ncol=2)
plt.ylabel(r"\% change in E[T] from guardrails")
plt.xlabel(r"System load ($\rho$)")
plt.xticks(ticks + len(new_order)/2 * bar_width, new_rhos)

plt.tight_layout()
#plt.savefig('many-bar.eps')

plt.close()

gs=[1, 1.25, 1.5, 2, 3, 4]
g_07 = """
0.700000,18.217127,22.332753,18.434788,23.214292,18.478221,20.125569
0.700000,18.225069,22.302179,18.426537,23.060931,18.501927,20.117076
0.700000,18.274696,22.500364,18.428391,23.663459,18.520698,20.212841
0.700000,18.499761,23.903021,18.443333,27.289560,18.567817,20.855956
0.700000,18.761383,24.745263,18.446044,29.719060,18.573785,21.086356
0.700000,18.935039,25.330434,18.442080,34.147721,18.524471,21.315051
""".strip().split("\n")

g_09 = """
0.900000,24.220618,28.556169,23.624321,30.430804,24.598187,25.516465
0.900000,24.268028,29.054900,23.424875,30.857499,24.639275,25.696355
0.900000,24.348860,29.293923,23.473226,31.208875,24.688852,25.756657
0.900000,25.839198,31.663813,23.663113,35.649953,25.307098,26.609737
0.900000,27.280629,33.123588,23.681733,39.658299,25.387388,26.887385
0.900000,28.456373,33.851120,23.569778,44.031456,25.514818,27.202624
""".strip().split("\n")
for i in order:
    times = [float(line.split(",")[i+1]) for line in g_09]
    plt.plot(gs, times, label="G-"+names[i], linewidth=4, color=colors[i])
plt.legend()
plt.xlabel("g")
plt.ylabel("Mean response time (E[T])")
plt.tight_layout()
#plt.savefig("gs.eps")
plt.close()

names="LWL,Random,JSQ,RR,JSQ-2,SITA-E".split(",")
bp = """
0.800000,36.033476,61.172848,34.570300,50.316724,42.699921,79.394615
0.980000,98.986495,124.488738,53.196507,103.078259,65.896013,559.743780
0.800000,34.114659,43.390677,34.645975,41.693679,38.554473,44.571183
0.980000,63.885670,72.250273,57.728419,68.669730,62.705196,77.550973
0.800000,34.561664,48.286602,34.728220,45.201511,40.544387,53.416495
0.980000,67.583790,82.739594,55.679617,75.928165,64.079265,109.639323
0.800000,35.102138,52.443527,34.723908,49.615154,41.558826,63.079127
0.980000,71.242538,91.453511,54.953986,82.166577,64.579479,168.525854
""".strip().split("\n")
bm = """
0.800000,113.243375,34.131625,18.272918,30.848574,22.888532,50.161628
0.980000,654.485730,94.545752,37.718017,90.541018,44.739911,283.250918
0.800000,17.905180,20.584921,18.114142,21.235026,19.264844,19.317870
0.980000,37.231965,41.705922,37.883433,41.151295,40.257759,39.108328
0.800000,17.641636,22.472912,18.254518,23.889640,20.431882,21.863166
0.980000,36.582673,45.495334,38.198427,46.271373,41.042334,41.402824
0.800000,18.701853,25.899142,18.515802,26.906306,21.576261,29.638957
0.980000,36.903979,52.690390,38.319748,53.757564,42.429494,46.744620
""".strip().split("\n")


gses = [[gs[2:4], gs[4:6], gs[6:8], gs[0:2]] for gs in [bp, bm]]
g_names = ["g=1", "g=2", "g=4","No guardrails"]
g_colors = ["orange", "green", "purple", "black"]
rho_names = ["80", "98"]
bar_width = 0.2
order = [5, 0, 1, 3, 4, 2]
ordered_names = [names[o] for o in order]
plot_names = ["bp", "bm"]
maxes = [[85, 200], [55, 150]]
for (plot_index, gs) in enumerate(gses):
    for (i, rho_name) in enumerate(rho_names):
        plt.figure(figsize=(6, 4.5))
        for j in range(len(gs)):
            g = gs[j]
            times = g[i].split(",")
            ordered_times = [float(times[o+1]) for o in order]
            ticks = np.arange(len(order))
            ax = plt.bar(ticks + bar_width * j, ordered_times, bar_width, color=g_colors[j],
                    label = g_names[j])
            rects = ax.patches
            for rect_index, rect in enumerate(rects):
                height = rect.get_height()
                if height > maxes[plot_index][i]:
                    plt.text(rect.get_x() + rect.get_width() * 1.2, maxes[plot_index][i]*0.93,
                            "{:.1f}".format(ordered_times[rect_index]))
        plt.ylim(ymax=maxes[plot_index][i])
        plt.legend()
        plt.ylabel("Mean response time (E[T])")
        plt.xlabel("Dispatching policy")
        plt.xticks(ticks + len(gs)/2 * bar_width, ordered_names)

        plt.tight_layout()
        plt.savefig('{}-many-pol-g-{}.eps'.format(plot_names[plot_index], rho_names[i]))
        plt.close()
