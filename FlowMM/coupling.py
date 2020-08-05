'''
...
'''

import torch
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform
from nflows.transforms import AffineCouplingTransform
from .angular import AnglePiecewiseRationalQuadraticCouplingTransform, DihedralPiecewiseRationalQuadraticCouplingTransform
from .coordinate import CoordinateTransform
from networkx import Graph

net = lambda in_features, out_features: ResidualNet(in_features, out_features, hidden_features=32)

class BondCouplingTransform(AffineCouplingTransform):
    '''
    Transform on bonds parameterized by angles and dihedrals
    '''

    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[: n - 1] = 1
        super().__init__(mask, net)

class AngleCouplingTransform(AnglePiecewiseRationalQuadraticCouplingTransform):
    '''
    Transform on angles parameterized by dihedrals and bonds
    '''

    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[n - 1 : 2 * n - 3] = 1
        super().__init__(mask, net)

class DihedralCouplingTransform(DihedralPiecewiseRationalQuadraticCouplingTransform):
    '''
    Transform on dihedrals parameterized by bonds and angles
    '''

    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[2 * n - 3 :] = 1
        super().__init__(mask, net)

class MMTransform(CompositeTransform):
    '''
    Transform that stacks B, A, D transforms in turn
    '''

    def __init__(self, graph: Graph, layers):
        n = graph.number_of_nodes()
        transforms = sum([[BondCouplingTransform(n), AngleCouplingTransform(n), DihedralCouplingTransform(n)] for _ in range(layers)], [CoordinateTransform(graph)])
        super().__init__(transforms)
