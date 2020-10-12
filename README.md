# FlowMM

Flow-based generative learning for molecular mechanics.

## Source Directory

- `FlowMM/coordinate.py` describes the coordinate transformation from cartesian to internal
- `FlowMM/distribution.py` describes the base distribution before flow transformation
- `FlowMM/angular.py` defines the coupling bijection transformation
- `FlowMM/contrastive.py` defines a class for performing contrastive learning

## Test Case

- `test/H2O2.py` provides a test case using an artificial force field that resembles an hydrogen peroxide molecule
