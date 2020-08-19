'''
Test case using an artificial force field that resembles hydrogen peroxide
'''

import json
from networkx import Graph
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam
from numpy import pi, random
from nflows.flows import Flow
from FlowMM import Atom, CoordinateTransform, MMConditionalTransform, MMAngularConditionalBaseDistribution, Contrastive, ForceField, ContextedDistribution, MMBondDistribution, MonteCarloSample, NaiveSample, IndependentTruncatedNormalDistribution

class H2O2ArtificialForceField(ForceField):
    '''
    '''

    def __init__(self, bondStiff=32, angleStiff=32, dihedralStiff=32):
        _parameter = lambda x: Parameter(torch.Tensor([x]))
        super().__init__()
        self.bondStiff = _parameter(bondStiff)
        self.angleStiff = _parameter(angleStiff)
        self.dihedralStiff = _parameter(dihedralStiff)
        self.logInversePartition = Parameter(torch.Tensor([6.573]))

    def forward(self, inputs: Tensor):
        bond = inputs[:, 0:3]
        angle = inputs[:, 3:5]
        dihedral = inputs[:, 5:6]
        return self.bondStiff * torch.sum((bond - 1.)**2, -1) + \
            self.angleStiff * torch.sum((angle - 2 * pi / 3)**2, -1) + \
            self.dihedralStiff * torch.sum(dihedral**2, -1)

H1 = Atom(0, 'H')
O1 = Atom(1, 'O')
O2 = Atom(2, 'O')
H2 = Atom(3, 'H')
H2O2 = Graph([(H1, O1), (O1, O2), (O2, H2)])
H2O2.graph['reference_atoms'] = (O1, H1, O2)
n = H2O2.number_of_nodes()

primitiveDistribution = IndependentTruncatedNormalDistribution(Tensor([1., 1., 1., 2 * pi / 3, 2 * pi / 3, 0]), torch.full([6], .125), torch.full([6], 6.))
internalSample = primitiveDistribution.sample(10000)
internalValidation = primitiveDistribution.sample(1000)
bondSample = internalSample[:, : n - 1]
angularSample = internalSample[:, n - 1 :]
bondValidation = internalValidation[:, : n - 1]
angularValidation = internalValidation[:, n - 1 :]

bondDistribution = MMBondDistribution(bondSample)
angularFlow = Flow(MMConditionalTransform(H2O2, layers=5), MMAngularConditionalBaseDistribution(H2O2))
angularFlowOptimizer = Adam(angularFlow.parameters(), lr=1e-3)

flowSteps = 1000
contrastiveSteps = 1000
batch = 1000

validationScore = -torch.mean(angularFlow.log_prob(angularValidation, bondValidation))

print('[Flow learning...]')

for epoch in range(flowSteps):
    angularFlowOptimizer.zero_grad()

    indices = random.choice(angularSample.shape[0], size=batch, replace=False)
    angularBatch = angularSample[indices]
    bondBatch = bondSample[indices]
    loss = -torch.mean(angularFlow.log_prob(angularSample, bondSample))
    loss.backward()
    angularFlowOptimizer.step()
    if epoch % (flowSteps // 10) == 0:
        with torch.no_grad():
            newValidationScore = -torch.mean(angularFlow.log_prob(angularValidation, bondValidation))
            print(f'Epoch: {epoch}, Loss: {loss.item():.5f}, Validation: {newValidationScore.item():.5f}')
            if newValidationScore < validationScore:
                validationScore = newValidationScore
            else:
                print('Stopped because of overfitting')
                break

flow = ContextedDistribution(bondDistribution, angularFlow, 4)
forcefield = H2O2ArtificialForceField(25, 30, 35)
contrastive = Contrastive(forcefield, flow)
contrastiveOptimizer = Adam(contrastive.parameters(), lr=1e-2)
noise = flow.sample(10000)

data = {
    'epochs': [],
    'losses': [],
    'parameters': [
        [], [], []
    ],
    'logInverseZ': []
}

print('[Contrastive learning...]')

for epoch in range(contrastiveSteps):
    contrastiveOptimizer.zero_grad()
    loss = contrastive(internalSample, noise, epoch)
    loss.backward(retain_graph=True)
    contrastiveOptimizer.step()
    if not epoch % 10:
        lossvalue = loss.item()
        parameters = [parameter.item() for parameter in list(contrastive.parameters())]
        data['epochs'].append(epoch)
        data['losses'].append(lossvalue)
        data['logInverseZ'].append(parameters[-1])
        for _ in range(3):
            data['parameters'][_].append(parameters[_])
        print(f"Epoch: {epoch}, Loss: {lossvalue:.5f}, Parameters: {parameters}")

json.dump([data, contrastive.data], open('data/H2O2.json', 'w'))
