# Communication scheme (Message Passing)

Remember our research question:
> "What is the impact of different levels of controller modularity on learning speed, coordination, and fault tolerance
> (e.g. amputations) in brittle-star-like robots trained with Reinforcement Learning?"

To test decentralized modularity (such as arm-level or segment-level controllers), the various modules *must* be able to
communicate with each other to achieve coordinated locomotion. This is accomplished through *message passing* in a Graph
Neural Network (GNN)-like architecture. Two prominent communication styles from the literature are N-step NerveNet (Wang
et al., 2018) and bottom-up top-down Shared Modular Policies (Huang et al., 2020).

We have chosen to apply **one uniform communication style** across all modular architectures, specifically opting for
**N-step NerveNet**.

## Rationale

Initially, our idea was to equip arm-level controllers with NerveNet message passing and segment-level controllers with
SMP. However, we evaluated that this introduces a threat to the validity of our research question. If we observe
differences in performance, it would be impossible to determine whether the variance is caused by the *level of
modularity*, or by the difference in the message passing scheme. To purely compare modularity, the communication scheme
style must remain constant.

Second, we decided that NerveNet is a better fit for our research. The morphology of our brittle star contains cycles at
the decentralized level (e.g., a ring of segments or arms around the body). NerveNet has proven to be robust for
arbitrary structures, including graphs with cycles. SMP inherently expects a tree structure for its bottom-up and
top-down pass. Applying SMP to a ring structure requires a workaround to break that cycle.

## Limitations and alternatives

Choosing NerveNet introduces a scalability issue as the morphology grows. In NerveNet, a message advances only one
segment or node per propagation step. When dealing with long arms (e.g., > 5 segments), this requires a large number of
propagation steps to transmit information from one tip of an arm to another.

If we would alternatively use SMP - which is possible - the inner states of nodes are shared across the entire graph in
just two passes. For very large or long morphologies, this would be much more scalable.

By rejecting SMP, we accept that our model might learn slower or require more computational power for highly segmented,
extended morphologies.

**References**

- Wang, Tingwu, Renjie Liao, Jimmy Ba, en S. Fidler. ‘NerveNet: Learning Structured Policy with Graph Neural Networks’. Conference paper presented bij International Conference on Learning Representations. 15 februari 2018. https://www.semanticscholar.org/paper/NerveNet:-Learning-Structured-Policy-with-Graph-Wang-Liao/249408527106d7595d45dd761dd53c83e5a02613.
- Huang, Wenlong, Igor Mordatch, en Deepak Pathak. ‘One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control’. arXiv:2007.04976. Preprint, arXiv, 9 juli 2020. https://doi.org/10.48550/arXiv.2007.04976.
