import numpy
from numpy import genfromtxt
import matplotlib.pyplot as plt



test = ['dgemm', 'sgemm']
platform =  ['legion_m2070', 'legion_k40',
             'emerald_k20m', 'emerald_k80m']
sub_position=[211, 212]

benchmark = [[j+'_'+i for j in platform] for i in test]
pos = 0



plt.suptitle("Benchmark: CUBLAS general matrix multiplication (single/double)",
             fontsize=14, fontweight='bold')

for tests  in benchmark:
    ax = plt.subplot(sub_position[pos])
    pos+=1
    ax.set_xlabel(r'problem size')
    ax.set_ylabel('GFlops')
    ax.set_xlim([1000,20000])
    for test in tests:
        benchmark_result = genfromtxt(test+'.txt', delimiter=' ')
        if test.startswith('legion_k40'):
            plt.plot(benchmark_result[:,0], benchmark_result[:,2],'-',
                     label=test, marker='D', markersize=8, linewidth=5)
        else:
            plt.plot(benchmark_result[:,0], benchmark_result[:,2],'-',
                     label=test, marker='D')

        legend = plt.legend(loc='upper left', prop={'size':10})

plt.show()
#plt.plot(benchmark_result)
