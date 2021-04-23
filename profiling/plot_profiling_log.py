import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


prof_data = []
with open('hierarchical_features.log', 'r') as f:
    for line in f:
        if line[:19] == 'INFO:root:profiling':
            prof_data.append(line[20:].rstrip())

print(prof_data)

list_of_times = []
times = None
feature_names = []
lc_lengths = []
n_curves = []
for line in prof_data:
    if line[:10] == 'Experiment':
        lc_len = int(line.split(' ')[3].rstrip(','))
        lc_lengths.append(lc_len)

        n = int(line.split(' ')[5])
        n_curves.append(n)
        if times is not None:
            list_of_times.append(np.array(times))
        times = []
        feature_names = []
        continue
    line_pieces = line.split(':')
    times.append(float(line_pieces[1]))
    feature_names.append(line_pieces[0].split('.')[-1].rstrip("'>"))
list_of_times.append(np.array(times))

list_of_times = np.stack(list_of_times)
total_times = list_of_times.sum(axis=1)[..., np.newaxis]
list_of_times = np.concatenate((list_of_times, total_times), axis=1)
feature_names.append('Total time')

lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

for i, name in enumerate(feature_names):
    plt.plot(lc_lengths, list_of_times[:, i]/n_curves, next(linecycler),
             label=name)

plt.legend(loc='lower left')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Light curve length')
plt.ylabel('Average time per light curve [s]')
plt.show()

# Periodo
print(feature_names[10])
plt.plot(lc_lengths, list_of_times[:, 10]/n_curves, '-',
         label='Period')

times = np.logspace(1, 3, 1000)
model = 0.0019 * times ** 0.79

plt.plot(times, model, '-', label='model')

plt.legend(loc='lower left')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Light curve length')
plt.ylabel('Average time per light curve [s]')
plt.show()

# Total
print(feature_names[-1])
plt.plot(lc_lengths, list_of_times[:, -1]/n_curves, '-',
         label='total time')

times = np.logspace(1, 3, 1000)
model = 0.02 * times ** 0.45

plt.plot(times, model, '-', label='model')

plt.legend(loc='lower left')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Light curve length')
plt.ylabel('Average time per light curve [s]')
plt.show()

# Periodo cada 5 detecciones
print(feature_names[10])
plt.plot(lc_lengths, list_of_times[:, 10]/n_curves, '-',
         label='Period')

times = np.logspace(1, np.log10(250), 1000)
model = 0.0019 * times ** 0.79
plt.plot(times, model, '-', label='normal model')

times = np.logspace(np.log10(250), 3, 1000)
model = 0.0019 / 5 * times ** 0.79
plt.plot(times, model, '-', label='trimmed model')

plt.legend(loc='lower left')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Light curve length')
plt.ylabel('Average time per light curve [s]')
plt.show()
