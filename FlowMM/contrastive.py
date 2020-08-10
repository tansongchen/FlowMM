'''
Implements the contrastive learning.
'''

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from nflows.flows import Flow

class ForceField(Module):
    def forward(self, inputs):
        raise NotImplementedError()

    def _forward_unimplemented(self, *inputs):
        return None

class Contrastive(Module):
    def __init__(self, potential: ForceField, flow: Flow):
        super().__init__()
        self.potential = potential
        self._flow = []
        self._flow.append(flow)

    def _log_ratio(self, u, epoch):
        logProbModel = self.potential.logInversePartition - self.potential(u)
        logProbNoise = self._flow[0].log_prob(u)
        if epoch % 100 == 0: print(logProbModel.mean(), logProbNoise.mean())
        return logProbModel - logProbNoise

    def forward(self, data: Tensor, noise: Tensor, epoch):
        nu = noise.shape[0] / data.shape[0]
        GData = self._log_ratio(data, epoch)
        GNoise = self._log_ratio(noise, epoch)
        if epoch % 100 == 0: print(GData.mean(), GNoise.mean())
        logLikelihoodData = torch.log(1 / (1 + torch.exp(-GData) * nu))
        logLikelihoodNoise = torch.log(1 / (1 + torch.exp(GNoise) / nu))
        return - logLikelihoodData.mean() - logLikelihoodNoise.mean() * nu

    def _forward_unimplemented(self, *inputs):
        return None
