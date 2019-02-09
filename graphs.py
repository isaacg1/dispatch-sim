import matplotlib.pyplot as plt

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
plt.plot(fcfs_random_rhos, fcfs_random, label='Random/FCFS', linewidth=4, color='orange')
plt.plot(fcfs_lwl_rhos, fcfs_lwl, label='LWL/FCFS', linewidth=4, color='cyan')
plt.plot(srpt_rhos, srpt_random, label='Random/SRPT', linewidth=4, color='red')
plt.plot(srpt_rhos, srpt_lwl, label='LWL/SRPT', linewidth=4, color='blue')

plt.ylim(ymax=500)

plt.legend(loc='upper center')
plt.ylabel("Mean response time ($E[T]$)")
plt.xlabel(r"System load ($\rho$)")
plt.tight_layout()
plt.savefig('srpt-vs-fcfs.eps', format='eps')

plt.close()

plt.figure(figsize=(5, 3.25))
plt.plot(srpt_rhos, srpt_random, label='Random/SRPT', linewidth=4, color='red')
plt.plot(srpt_rhos, srpt_lwl, label='LWL/SRPT', linewidth=4, color='blue')
plt.plot(guard_rhos, guard_random, label='G-Random/SRPT', linewidth=4, color='red', linestyle='dashed')
plt.plot(guard_rhos, guard_lwl, label='G-LWL/SRPT', linewidth=4, color='blue', linestyle='dashed')

plt.ylim(ymax=100)

plt.legend(loc='upper left')
plt.ylabel("Mean response time (E[T])")
plt.xlabel(r"System load ($\rho$)")
plt.tight_layout()
plt.savefig('guard-vs-unguard.eps', format='eps')

plt.close()

names="LWL,Random,JSQ,RR,JIQ,JSQ-2".split(",")
colors="red,blue,brown,purple,green,black".split(",")
order=[2,1,4,0,5,3]
big_guard = """
0.020000,16.322658,16.476013,16.322658,16.378918,16.322658,16.326238
0.100000,17.102942,17.939078,17.102942,17.761188,17.102942,17.194066
0.200000,16.969325,18.462286,16.969325,19.151986,16.969325,17.286224
0.300000,16.921320,19.202588,16.926574,20.710580,16.930679,17.652157
0.400000,17.118581,20.277018,17.143213,22.408309,17.144732,18.396778
0.500000,17.114201,21.045469,17.213044,23.779406,17.197612,18.787875
0.600000,17.482834,22.202206,17.583614,25.121859,17.617069,19.678541
0.700000,18.499761,23.903021,18.443333,27.289560,18.567817,20.855956
0.750000,19.327527,25.090540,19.084351,28.231669,19.292106,21.535127
0.800000,20.652563,26.515552,20.031590,30.075016,20.515526,22.773465
0.850000,22.811730,28.840072,21.639697,32.195965,22.483604,24.361761
0.900000,25.839198,31.663813,23.663113,35.649953,25.307098,26.609737
0.925000,28.041328,33.792777,25.219335,39.106187,27.337755,28.239603
0.950000,30.664572,36.516256,27.109098,42.732526,30.335352,30.311179
0.970000,33.603007,39.501260,29.441802,45.863687,33.552967,32.916480
0.980000,35.600782,42.302914,31.022293,50.848657,35.965848,34.739854
0.990000,37.842586,46.049668,33.165031,57.843006,39.178987,36.979235
""".strip().split("\n")
big_unguard = """
0.020000,16.322658,16.513919,16.322658,16.379736,16.322658,16.326238
0.100000,17.102942,17.996931,17.102942,17.800894,17.102942,17.189182
0.200000,16.969325,18.828510,16.969325,18.395036,16.969325,17.333735
0.300000,16.921320,20.071873,16.926545,19.246928,16.930790,17.782272
0.400000,17.118517,21.902034,17.143072,20.724927,17.144868,18.662175
0.500000,17.123442,23.707809,17.210141,22.151332,17.193264,19.412054
0.600000,17.667326,26.239961,17.588328,24.392763,17.594538,20.592212
0.700000,19.501375,29.513937,18.383275,27.479982,18.517302,22.276180
0.750000,21.176340,31.598692,18.974438,29.546273,19.214791,23.283792
0.800000,23.915872,34.829777,19.907702,32.632625,20.366231,24.578865
0.850000,28.913315,39.513429,21.295491,37.450798,22.469923,26.466192
0.900000,35.277139,46.227297,23.222403,45.781808,25.061370,28.810970
0.925000,39.439541,51.384034,24.504264,51.986210,27.297853,30.461680
0.950000,44.780032,58.885740,26.372493,60.254401,30.430094,32.790417
0.970000,50.375291,67.008844,28.517590,71.028522,34.523204,35.285860
0.980000,53.977575,73.187361,30.010294,80.263894,38.216056,37.321719
0.990000,57.105689,87.738841,32.380991,133.180350,42.894850,39.350937
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
plt.savefig('many-policies.eps')

plt.close()

import numpy as np

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
plt.ylabel("% change in E[T] from guardrails")
plt.xlabel(r"System load ($\rho$)")
plt.xticks(ticks + len(new_order)/2 * bar_width, new_rhos)

plt.tight_layout()
plt.savefig('many-bar.eps')

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
plt.savefig("gs.eps")
