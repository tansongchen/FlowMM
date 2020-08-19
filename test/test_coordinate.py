'''
Test the CoordinateTransform class
'''

from unittest import TestCase
from networkx import Graph
from torch import Tensor
from numpy import log, sin
from FlowMM import CoordinateTransform, Atom

class TestCoordinate(TestCase):
    '''
    Create a H2O2 molecule and convert between cartesian and internal coordinates
    '''

    def test_basic(self):
        H1 = Atom(0, 'H')
        O1 = Atom(1, 'O')
        O2 = Atom(2, 'O')
        H2 = Atom(3, 'H')
        H2O2 = Graph([(H1, O1), (O1, O2), (O2, H2)])
        H2O2.graph['reference_atoms'] = (O1, H1, O2)

        coordinate = CoordinateTransform(H2O2)

        internal = Tensor([[1.1, 1.2, 1.3, 2.4, 2.5, 3.0]])
        expt = log(1.1**2 * sin(2.4) * 1.3**2)
        cartesian, logabsdet = coordinate.inverse(internal)
        internalAgain, neglogabsdet = coordinate.forward(cartesian)
        self.assertAlmostEqual((internal - internalAgain).norm(), 0, delta=1e-4)
        self.assertAlmostEqual(logabsdet, expt, delta=1e-4)
        self.assertAlmostEqual(neglogabsdet, -expt, delta=1e-4)
