'''
...
'''

import torch
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform, AffineCouplingTransform, RandomPermutation
from .angular import AnglePiecewiseRationalQuadraticCouplingTransform, DihedralPiecewiseRationalQuadraticCouplingTransform
from .coordinate import CoordinateTransform
from networkx import Graph

net = lambda ins, outs: ResidualNet(ins, outs, 32)

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

class AngularCouplingTransform(DihedralPiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, features, contexts, hiddens, i):
        mask = create_alternating_block_binary_mask(features, even=(i % 2 == 0), block_length=2)
        residualNet = lambda ins, outs: ResidualNet(ins, outs, hiddens, context_features=contexts, dropout_probability=0.25)
        super().__init__(mask, residualNet)

class MMConditionalTransform(CompositeTransform):
    '''
    Using bond to parameterize angle and dihedral
    '''

    def __init__(self, graph: Graph, layers, hiddens=32):
        n = graph.number_of_nodes()
        features = 2 * n - 5
        contexts = n - 1

        super().__init__(
            sum([[AngularCouplingTransform(features, contexts, hiddens, i), RandomPermutation(features)] for i in range(layers)], [RandomPermutation(features)])
        )

def create_alternating_block_binary_mask(features, even=True, block_length=1):
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()

    if even:
        for i in range(block_length):
            mask[i::(2*block_length)] += 1
    else:
        for i in range(block_length, 2*block_length):
            mask[i::(2*block_length)] += 1    

    return mask