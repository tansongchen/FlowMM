'''
Transforming Cartesian coordinates to internal coordinates
'''

from nflows.transforms import Transform
from networkx import Graph
import torch

class Atom:
    '''
    Data structure consisting of index and graph infomation
    '''
    def __init__(self, index, label):
        self.index = index
        self.label = label
        self.visited = False
        self.predecessor = None
        self.parents = None

    def __repr__(self):
        return f'{self.index}: {self.label}'

class CoordinateTransform(Transform):
    '''
    Transform between Cartesian coordinates and internal coordinates
    '''

    def __init__(self, graph: Graph):
        '''
        Initializing using the molecular graph
        '''
        super().__init__()
        self.graph = graph
        self.atom1, self.atom2, self.atom3 = self.graph.graph['reference_atoms']
        self.atom2.predecessor = self.atom1
        self.atom3.predecessor = self.atom1
        self.atom1.predecessor = self.atom2
        self.atom1.visited = True
        self.atom2.visited = True
        self.atom3.visited = True

        self.visited = []
        queue = [self.atom1, self.atom2, self.atom3]

        while queue:
            atom = queue.pop(0)
            for neighbor in graph[atom]:
                if not neighbor.visited:
                    self.visited.append(neighbor)
                    neighbor.predecessor = atom
                    queue.append(neighbor)
                    neighbor.visited = True
                    if atom in (self.atom1, self.atom2):
                        neighbor.parents = (self.atom3.index, atom.predecessor.index, atom.index)
                    else:
                        neighbor.parents = (atom.predecessor.predecessor.index, atom.predecessor.index, atom.index)

    @staticmethod
    def _normalize(t):
        return t / torch.norm(t, dim=-1, keepdim=True)

    @staticmethod
    def _innerProduct(t, u):
        return torch.sum(t * u, dim=-1)

    @staticmethod
    def _logabsdet(bond, angle):
        return torch.sum(torch.log(torch.abs(
            bond[:, :-2]**2*torch.sin(angle[:, :-1])
            )), -1) + torch.log(torch.abs(bond[:, -1]**2))

    def forward(self, cartesian, context=None):
        '''
        Convert from Cartesian coordinates to internal coordinates
        '''
        N = cartesian.shape[0]
        n = self.graph.number_of_nodes()
        internal = torch.Tensor(N, 3 * n - 6)
        bond = internal[:, 0 : n - 1]
        angle = internal[:, n - 1: 2 * n - 3]
        dihedral = internal[:, 2 * n - 3 : 3 * n - 6]

        r21 = cartesian[:, self.atom2.index, :] - cartesian[:, self.atom1.index, :]
        r31 = cartesian[:, self.atom3.index, :] - cartesian[:, self.atom1.index, :]
        u21 = self._normalize(r21)
        u31 = self._normalize(r31)

        bond[:, -2] = torch.norm(r21, dim=-1)
        bond[:, -1] = torch.norm(r31, dim=-1)
        angle[:, -1] = torch.acos(self._innerProduct(u21, u31))

        # https://en.wikipedia.org/wiki/Dihedral_angle
        for index, atom in enumerate(self.visited):
            (i, j, k), l = atom.parents, atom.index
            rIJ = cartesian[:, i, :] - cartesian[:, j, :]
            rJK = cartesian[:, j, :] - cartesian[:, k, :]
            rLK = cartesian[:, l, :] - cartesian[:, k, :]
            uJK = self._normalize(rJK)
            uLK = self._normalize(rLK)
            nIJJK = torch.cross(rIJ, rJK)
            nLKJK = torch.cross(rLK, rJK)
            nn = torch.cross(nLKJK, nIJJK)
            y = self._innerProduct(rJK, nn)
            x = torch.norm(rJK, dim=-1) * self._innerProduct(nIJJK, nLKJK)
            bond[:, index] = torch.norm(rLK, dim=-1)
            angle[:, index] = torch.acos(self._innerProduct(uJK, uLK))
            dihedral[:, index] = torch.atan2(y, x)
        return internal, -self._logabsdet(bond, angle)

    def inverse(self, internal, context=None):
        '''
        Convert from internal coordinates to Cartesian coordinates
        '''

        N = internal.shape[0]
        n = self.graph.number_of_nodes()
        bond = internal[:, 0 : n - 1]
        angle = internal[:, n - 1: 2 * n - 3]
        dihedral = internal[:, 2 * n - 3 : 3 * n - 6]
        cartesian = torch.zeros(N, n, 3)

        cartesian[:, self.atom2.index, 2] = bond[:, -2]
        cartesian[:, self.atom3.index, 1] = bond[:, -1] * torch.sin(angle[:, -1])
        cartesian[:, self.atom3.index, 2] = bond[:, -1] * torch.cos(angle[:, -1])

        for index, atom in enumerate(self.visited):
            (i, j, k), l = atom.parents, atom.index
            rJK = cartesian[:, j, :] - cartesian[:, k, :]
            z = self._normalize(rJK)
            rIJ = cartesian[:, i, :] - cartesian[:, j, :]
            y = self._normalize(torch.cross(rIJ, rJK, dim=-1))
            x = torch.cross(z, y, dim=-1)
            u = torch.sin(angle[:, index:index+1]) * torch.cos(dihedral[:, index:index+1]) * x + torch.sin(angle[:, index:index+1]) * torch.sin(dihedral[:, index:index+1]) * y + torch.cos(angle[:, index:index+1]) * z
            cartesian[:, l, :] = cartesian[:, k, :] + bond[:, index:index+1] * u
        return cartesian, self._logabsdet(bond, angle)
