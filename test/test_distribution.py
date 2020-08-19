'''
Test the CoordinateTransform class
'''

from unittest import TestCase
import torch
from torch import Tensor
from numpy import log, sin
from FlowMM import (
    IndependentUniformDistribution,
    IndependentNormalDistribution,
    IndependentTruncatedNormalDistribution
)

class TestDistribution(TestCase):
    '''
    test
    '''

    def test_ITND(self):
        mean = torch.full([10], 1.)
        std = torch.full([10], 2.)
        maxdev = torch.full([10], 5.)
        itnd = IndependentTruncatedNormalDistribution(mean, std, maxdev)
        sample = itnd.sample(100000)
        self.assertAlmostEqual(torch.mean(sample), mean.mean(), delta=1e-2)
        self.assertAlmostEqual(torch.std(sample, dim=0).mean(), std.mean(), delta=1e-2)
