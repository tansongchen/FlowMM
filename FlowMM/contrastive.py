'''
Implements the contrastive learning.
'''

import torch
from torch import Tensor
from torch.nn import Module
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
        self.data = {
            'model@data': [],
            'model@noise': [],
            'flow@data': [],
            'flow@noise': []
        }

    def _log_ratio(self, u, epoch, label):
        logProbModel = self.potential(u)
        logProbFlow = self._flow[0].log_prob(u)
        if not epoch % 100:
            self.data[f'model{label}'].append(logProbModel.mean().item())
            self.data[f'flow{label}'].append(logProbFlow.mean().item())
        return logProbModel - logProbFlow

    def forward(self, data: Tensor, noise: Tensor, epoch):
        nu = noise.shape[0] / data.shape[0]
        GData = self._log_ratio(data, epoch, '@data')
        GNoise = self._log_ratio(noise, epoch, '@noise')
        logLikelihoodData = torch.log(1 / (1 + torch.exp(-GData) * nu))
        logLikelihoodNoise = torch.log(1 / (1 + torch.exp(GNoise) / nu))
        return - logLikelihoodData.mean() - logLikelihoodNoise.mean() * nu

    def _forward_unimplemented(self, *inputs):
        return None
