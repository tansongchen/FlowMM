'''
Combined base distribution
'''

import torch
from numpy import pi
from nflows.distributions.base import Distribution
from nflows.distributions import DiagonalNormal
from nflows.distributions.uniform import BoxUniform

class MMBaseDistribution(Distribution):
    '''
    Base distribution for bonds, angles and dihedrals corresponding to a naive additive force field
    '''

    def __init__(self, graph):
        super().__init__()
        n = graph.number_of_nodes()
        self.n = n
        self.bond = DiagonalNormal([n - 1])
        self.angle = BoxUniform(low=torch.zeros([n - 2]), high=torch.full([n - 2], pi))
        self.dihedral = BoxUniform(low=torch.full([n - 3], -pi), high=torch.full([n - 3], pi))

    def _log_prob(self, inputs, context=None):
        bond, angle, dihedral = inputs
        return self.bond.log_prob(bond) + self.angle.log_prob(angle) + self.dihedral.log_prob(dihedral)

    def _sample(self, num_samples, context=None):
        n = self.n
        internal = torch.Tensor(num_samples, 3 * n - 6)
        internal[:, : n - 1] = self.bond.sample(num_samples, context)
        internal[:, n - 1 : 2 * n - 3] = self.angle.sample(num_samples, context)
        internal[:, 2 * n - 3 :] = self.dihedral.sample(num_samples, context)
        return internal
