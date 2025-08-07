# Differentiable Programming Handbook

Differentiable implementation of common computer science algorithms.

> *Trying out the Tensorflow 2.0 Gradient tape during the pandemic*

## Motivation

These algorithms can be thought as overengineered loss functions. They are meant to be used in conjuction with deep learning networks. However, unlike loss function, they dont need to be attached to the end of the network. As demonstrated in the [Bubble Sort](notebooks/bubble-sort.ipynb) example, it can also be interleaved with the graph of the network.

All algorithms in this repository follow these rules

1. **It must be deterministic in forward pass** - Although, these are intended to be used in conjunction with deep neural networks, there must be a very clear boundary of separation of code with learnable parameters and code without learnable parameters. Therefore no stochasticity is allowed, either during run time or during the generation of code which will be executed at runtime.
2. **It must be lossless in forward pass** - The algorithms must behave identically to classical algorithms when the inputs are discrete. When non-discrete data points are passed, it should produce well behaved, interpretable and continious output.
3. **It must have well definied gradients in backward pass** - The algorithms should have well defined gradients with at least one of its inputs.

## Contents

* [Bubble Sort](notebooks/bubble-sort.ipynb) - Differentiable implementation of bubble sort with configurable (learnable) comparator function
* [Boolean Satisfiability](notebooks/boolean-satisfiability.ipynb) - Solving CNF boolean SAT with gradient descent
* [Subset Sum](notebooks/subset-sum.ipynb) - Solving Subset Sum with gradient descent
* [Higher Order Functions](notebooks/higher-order-functions.ipynb) - Demonstrating gradient descent through higher order functions
* [Differentiable indexed arrays](notebooks/differentiable-indexed-arrays.ipynb) - Listing strategies to implement array indexing with gradients defined for both input and index.
* [Stacks](notebooks/stacks.ipynb) - A basic implementation of differentiable stacks
* [Differentiable Indirection](notebooks/differentiable-indirection.ipynb) - Implementation of circular linked lists with differentiable indexing
* [Custom gradient](notebooks/custom-gradient.ipynb) - Demonstration of tensorflow custom gradients
* [Pathfinding](notebooks_torch/pathfinding_dense.ipynb) - Pathfinding in an adjacency matrix with backprop
