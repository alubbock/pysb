""" Run the Extrinsic Apoptosis Reaction Model (EARM) using StochKit's
stochastic simulation algorithm (SSA) implementation.
"""

import matplotlib.pyplot as plt
import numpy as np

from pysb.examples.earm_1_0 import model
from pysb.simulator import StochKitSimulator
import os


def plot_mean_min_max(name, title=None):
    x = np.array([tr[:][name] for tr in trajectories]).T
    if not title:
        title = name
    plt.figure(title)
    plt.plot(tout.T, x, '0.5', lw=2, alpha=0.25) # individual trajectories
    plt.plot(tout[0], x.mean(1), 'k--', lw=3, label="Mean")
    plt.plot(tout[0], x.min(1), 'b--', lw=3, label="Minimum")
    plt.plot(tout[0], x.max(1), 'r--', lw=3, label="Maximum")
    plt.legend(loc=0)
    plt.xlabel('Time')
    plt.ylabel('Population of %s' % name)


num_runs = 20

# Only run twice for PySB's tests, for speed. PySB users can ignore this.
if os.environ.get('PYSB_TESTS', None):
    num_runs = 2

tspan = np.linspace(0, 20000, 1000)
sim = StochKitSimulator(model, tspan)
simres = sim.run(n_runs=num_runs, seed=None, algorithm="ssa")

trajectories = simres.all
tout = simres.tout

plot_mean_min_max('Bid_unbound')
plot_mean_min_max('PARP_unbound')
plot_mean_min_max('mSmac_unbound')
plot_mean_min_max('tBid_total')
plot_mean_min_max('CPARP_total')
plot_mean_min_max('cSmac_total')

plt.show()
