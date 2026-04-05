# noam-micrograd

A from-scratch implementation of a scalar-valued autograd engine and a small neural network library. This project is part of a deep-dive study into the foundations of deep learning, following the curriculum of Andrej Karpathy's Zero to Hero series.

## Technical Objectives
The goal of this lab is to rebuild the core mechanics of backpropagation and neural modeling without relying on high-level frameworks.
* Implement a Scalar Autograd Engine with a full suite of mathematical operations.
* Build a Dynamic Computation Graph that supports automated gradient calculation.
* Construct a Neural Network API (Neurons, Layers, MLPs) on top of the engine.
* Verify learning through optimization on toy datasets.

## Tech Stack
* Language: Python 3.x
* Environment: Jupyter Lab / VS Code
* Key Concepts: Chain Rule, Partial Derivatives, Topological Sorting, Stochastic Gradient Descent (SGD).

## Implementation Status
* [x] Initial Repository Setup
* [ ] Value class core (data, grad, op tracking)
* [ ] Basic arithmetic operators (+, *, -, /, **)
* [ ] Activation functions (tanh, relu)
* [ ] Backward pass logic (Recursive & Topological Sort)
* [ ] Module API (Neuron, Layer, MLP)
* [ ] Model training & loss visualization

## Engineering Journal
*Document specific hurdles or architectural decisions here.*
* Topological Sorting: Gradients must be computed in a specific order to ensure all dependencies are resolved before a node's gradient is finalized.
* Gradient Accumulation: Using += for gradient updates to handle nodes used multiple times in the computation graph.

## Project Structure
* /notebooks: Exploratory coding and step-by-step lecture follow-along.
* /src: Refactored implementation of the engine.
* requirements.txt: Minimal dependencies (e.g., graphviz for visualization).

## References
This implementation is based on the foundations laid out in the micrograd project by Andrej Karpathy.
