'''
...
'''

import torch
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform
from nflows.transforms import AffineCouplingTransform, PiecewiseRationalQuadraticCouplingTransform
from .angular import AnglePiecewiseRationalQuadraticCouplingTransform, DihedralPiecewiseRationalQuadraticCouplingTransform
from .ic import InternalCoordinateTransform
from networkx import Graph

net = lambda in_features, out_features: ResidualNet(in_features, out_features, hidden_features=32)

class BondCouplingTransform(AffineCouplingTransform):
    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[: n - 1] = 1
        super().__init__(mask, net)

class AngleCouplingTransform(AnglePiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[n - 1 : 2 * n - 3] = 1
        super().__init__(mask, net)

class DihedralCouplingTransform(DihedralPiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, n):
        mask = torch.zeros(3 * n - 6)
        mask[2 * n - 3 :] = 1
        super().__init__(mask, net)

class BADCompositeTransform(CompositeTransform):
    def __init__(self, graph: Graph, layers):
        n = graph.number_of_nodes()
        transforms = sum([[BondCouplingTransform(n), AngleCouplingTransform(n), DihedralCouplingTransform(n)] for _ in range(layers)], [InternalCoordinateTransform(graph)])
        super().__init__(transforms)
