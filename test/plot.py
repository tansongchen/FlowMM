import json
import matplotlib
import matplotlib.pyplot as plt

data, cdata = json.load(open('data/H2O2.json'))
epochs = data['epochs']

plt.rcParams['savefig.dpi'] = 600 
matplotlib.rcParams.update({'font.size': 16})
figure, (parameters) = plt.subplots(1, 1, sharex=True)
# figure, (parameters, loss, crossentropy, ce2) = plt.subplots(4, 1, sharex=True)

parameters.plot(epochs, data['logInverseZ'], label=r'$-\log Z$', color='purple')
parameters.set_xlim((0, 1000))
parameters.set_ylim((6, 7))
parameters.plot(epochs, [6.963 for _ in epochs], label=r'Exact $-\log Z$', color='black')
parameters.legend(fontsize=10)
parameters.set_title('(a)')

ff = parameters.twinx()
names = [r'$k_{\rm bond}$', r'$k_{\rm angle}$', r'$k_{\rm dihedral}$']
for index, entry in enumerate(data['parameters']):
    ff.plot(epochs, entry, label=names[index])
ff.plot(epochs, [32 for _ in epochs], label=r'Exact $k$')

ff.set_ylim((20, 40))
ff.legend(fontsize=10)

# loss.plot(epochs, data['losses'], label='Loss')
# loss.set_xlim((0, 1000))
# loss.set_ylim((1.3, 1.4))
# loss.legend(fontsize=10)
# loss.set_title('(b)')

parameters.set_xlabel('Epoch')

# for name, entry in cdata.items():
#     crossentropy.plot(epochs, entry, label=name)

# for name, entry in cdata.items():
#     ce2.plot(epochs, entry, label=name)

# crossentropy.set_xlim((0, 1000))
# crossentropy.set_ylim((3.85, 4.05))
# crossentropy.legend(fontsize=10)
# crossentropy.set_title('(c)')

# ce2.set_ylim(0.95, 1.15)  # most of the data

# crossentropy.spines['bottom'].set_visible(False)
# ce2.spines['top'].set_visible(False)
# crossentropy.xaxis.tick_top()
# ce2.tick_params(labeltop=False)
# ce2.xaxis.tick_bottom()

# d = .015
# kwargs = dict(transform=crossentropy.transAxes, color='k', clip_on=False)
# crossentropy.plot((-d, +d), (-d, +d), **kwargs)
# crossentropy.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# kwargs.update(transform=ce2.transAxes)
# ce2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ce2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
# ce2.set_xlabel('Epoch')

plt.show()
