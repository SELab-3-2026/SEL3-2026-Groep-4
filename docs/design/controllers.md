# Levels of modularity and topology

The brittle star can be controlled at different levels. A monolithic controller processes all inputs and outputs at
once, whereas modular controllers divide the brains across the body, inspired by the biology of brittle stars.

We define four architectures to compare:

1. **Centralized, monolithic**: A single Multi Layer Perceptron per robot that receives all observations and outputs
all actions.
2. **Fully connected arm-level**: Each arm contains an MLP that processes the inputs for that arm, an MLP that processes
the communicated inner-states, and an MLP that outputs the actions for that arm. One policy for these MLPs is shared
across the arms. The controllers in each arm are connected to each other and form a fully connected graph. There is no
central disk, but the controllers are fully connected.
3. **Ring arm-level**: Identical setup to the fully connected arm-level, but the controllers are connected in a ring
structure. This setup is considered less centralized than the fully connected graph.
4. **Segment-level**: Each segment contains the three MLPs discussed above. The base segments, attached to the body,
form a ring structure, with the remaining segments attached as extended "strings". Segments can only communicate with
segments that are physically connected to it.

## Rationale

To fairly compare decentralized modularity against centralized control, the decentralized models should not be allowed
to contain a central organ acting as a bottleneck or coordinator. By removing the central disk in the decentralized
models and replacing it with a ring topology, we closely approximate the biological reality of the brittle star and test
a decentralized morphology.

The fully connected graph functions as an intermediate step in between a fully centralized and a decentralized ring. We
use it to test whether our models scale to more complex structures.
