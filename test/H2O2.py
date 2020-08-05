'''
Test case using an artificial force field that resembles hydrogen peroxide
'''

from networkx import Graph
import torch
from nflows.flows import Flow
from FlowMM import Atom, MMBaseDistribution, MMTransform

H2O2 = Graph()

H1 = Atom(0, 'H')
O1 = Atom(1, 'O')
O2 = Atom(2, 'O')
H2 = Atom(3, 'H')

H2O2.add_nodes_from([H1, O1, O2, H2])
H2O2.add_edges_from([(H1, O1), (O1, O2), (O2, H2)])
H2O2.graph['reference_atoms'] = (O1, H1, O2)

mm = MMTransform(H2O2, 1)
base = MMBaseDistribution(H2O2)
flow = Flow(mm, base)

cart = torch.Tensor([
    [
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.]
    ]
])

i, l = mm.forward(cart)
c, l = mm.inverse(i)
