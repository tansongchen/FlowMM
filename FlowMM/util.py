'''
2D potentials and sampling with Monte Carlo
'''

from random import random
from numpy import pi
from nflows.distributions import Distribution
import torch

def MonteCarloSample(potential, N, n):
    '''
    Simple sampling
    '''
    samples = torch.Tensor(N, 3 * n - 6)
    cartesian = torch.Tensor([[1., 1., 1., 2 * pi / 3, 2 * pi / 3, 0.]])
    u = potential(cartesian)
    randomMove = lambda: (random() - 0.5) * 0.1
    nInit = 1000
    space = 10
    for nStep in range(-nInit, N * space):
        index = int(random() * (3 * n - 6))
        xTrial = cartesian.clone().detach()
        xTrial[0, index] += randomMove()
        uTrial = potential(xTrial)
        if random() < min(1., torch.exp(u - uTrial)):
            xTrial[:, 2 * n - 3 :] -= (xTrial[:, 2 * n - 3 :] / pi).type(torch.IntTensor) * pi
            cartesian = xTrial
            u = uTrial
        if nStep >= 0 and nStep % space == 0:
            iSample = nStep // space
            samples[iSample, ...] = cartesian
    return samples

def NaiveSample(N, n, bond: Distribution, angle: Distribution, dihedral: Distribution):
    internal = torch.Tensor(N, 3 * n - 6)
    internal[:, 0 : n - 1] = bond.sample(N)
    internal[:, n - 1 : 2 * n - 3] = angle.sample(N)
    internal[:, 2 * n - 3 : 3 * n - 6] = dihedral.sample(N)

if __name__ == '__main__':
    from FlowMM import ForceField
    class Diatomic(ForceField):
        def __init__(self, stiff=100):
            super().__init__()
            self.stiff = stiff

        def forward(self, inputs):
            return self.stiff * torch.sum((torch.norm(inputs[:, 0, :] - inputs[:, 1, :], dim=-1) - 1)**2)

    diatomic = Diatomic()
    sample = MonteCarloSample(diatomic, 1000, 2)
    bonds = torch.norm(sample[:, 0, :] - sample[:, 1, :], dim=-1)
    print(bonds.mean(), bonds.std())
