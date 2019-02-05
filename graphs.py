import matplotlib.pyplot as plt

"""
S=Bimodal(1, 1000, 0.9995)
k=10
FCFS
SRPT
Guard
c=1+1/(ln(1/(1-rho)))
g=4

rho,lwl,random
"""
fcfs_lwl_data = """\
0.020000,12.995202
0.100000,14.928611
0.200000,14.620217
0.300000,15.301881
0.400000,15.628377
0.500000,16.403296
0.600000,25.242105
0.700000,65.991877
0.750000,126.612914
0.800000,229.907886
0.850000,437.730488
0.900000,919.522753
0.925000,1382.443129
0.950000,2525.116708
0.970000,4325.164153
0.980000,5549.203726
0.990000,7273.984542\
"""
fcfs_random_data = """\
0.020000,46.167003
0.040000,93.839237
0.060000,126.707641
0.080000,160.137586
0.100000,205.595807
0.120000,247.391221
0.140000,289.629582
0.160000,330.176192
0.180000,385.423438
0.200000,443.231816
0.220000,510.732966
0.240000,572.076196
0.260000,626.870387\
"""
srpt = """\
0.020000,12.995202,13.110015
0.100000,14.928611,15.810471
0.200000,14.620217,16.536597
0.300000,15.301881,18.684220
0.400000,15.628276,20.588345
0.500000,16.270770,22.057472
0.600000,21.765819,24.515265
0.700000,43.660215,27.758991
0.750000,67.266796,29.816389
0.800000,110.482724,33.243512
0.850000,183.003661,39.187186
0.900000,280.190651,47.663144
0.925000,363.163848,52.592527
0.950000,487.599986,59.496319
0.970000,565.559590,67.283348
0.980000,632.026693,71.943615
0.990000,634.675484,76.933911\
"""
guard = """\
0.020000,12.995202,13.110015
0.100000,15.238229,15.799149
0.200000,15.254834,16.515774
0.300000,16.568977,18.615491
0.400000,17.586180,20.399080
0.500000,17.954771,21.539851
0.600000,18.555058,23.443406
0.700000,19.494621,25.875626
0.750000,20.135688,26.754865
0.800000,20.792493,29.485870
0.850000,21.995730,33.169145
0.900000,23.619247,37.954372
0.925000,24.846277,41.007224
0.950000,27.001974,42.647772
0.970000,29.856300,46.996720
0.980000,31.894272,49.713743
0.990000,34.558030,52.420628\
"""

fcfs_lwl_rhos, fcfs_lwl =\
    [[float(line.split(',')[i]) for line in fcfs_lwl_data.split()] for i in range(2)]

fcfs_random_rhos, fcfs_random =\
    [[float(line.split(',')[i]) for line in fcfs_random_data.split()] for i in range(2)]

srpt_rhos, srpt_lwl, srpt_random =\
    [[float(line.split(',')[i]) for line in srpt.split()] for i in range(3)]

guard_rhos, guard_lwl, guard_random =\
    [[float(line.split(',')[i]) for line in guard.split()] for i in range(3)]

plt.figure(figsize=(8, 5))
plt.plot(fcfs_random_rhos, fcfs_random, label='Random/FCFS', linewidth=4)
plt.plot(fcfs_lwl_rhos, fcfs_lwl, label='LWL/FCFS', linewidth=4)
plt.plot(srpt_rhos, srpt_random, label='Random/SRPT', linewidth=4)
plt.plot(srpt_rhos, srpt_lwl, label='LWL/SRPT', linewidth=4)

plt.ylim(ymax=500)

plt.legend(loc='upper center')
plt.title("$E[T]$ under FCFS and SRPT scheduling")
plt.ylabel("Mean response time ($E[T]$)")
plt.xlabel(r"System load ($\rho$)")
plt.savefig('srpt-vs-fcfs.eps', format='eps')

plt.close()

plt.figure(figsize=(8, 5))
plt.plot(srpt_rhos, srpt_random, label='Random/SRPT', linewidth=4)
plt.plot(srpt_rhos, srpt_lwl, label='LWL/SRPT', linewidth=4)
plt.plot(guard_rhos, guard_random, label='G-Random/SRPT', linewidth=4)
plt.plot(guard_rhos, guard_lwl, label='G-LWL/SRPT', linewidth=4)

plt.ylim(ymax=100)

plt.legend(loc='upper center')
plt.title("$E[T]$ with and without guardrails")
plt.ylabel("Mean response time (E[T])")
plt.xlabel(r"System load ($\rho$)")
plt.savefig('guard-vs-unguard.eps', format='eps')
