# MIT License

# Copyright (c) 2020 Simon Schug, João Sacramento

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from lib import cost


def create_activations(name, n_layers):
    """
    Create  activation functions for every layer of the network.

    Args:
        name: Name of the activation function
        n_layers: Number of layers

    Returns:
        List of activation functions for every layer
    """
    if name == 'relu':
        phi_l = torch.relu
    elif name == "leaky_relu":
        def phi_l(x): torch.nn.functional.leaky_relu(x, negative_slope=0.05)
    elif name == 'softplus':
        phi_l = torch.nn.functional.softplus
    elif name == 'sigmoid':
        phi_l = torch.sigmoid
    elif name == 'hard_sigmoid':
        def phi_l(x): torch.clamp(x, min=0, max=1)
    else:
        raise ValueError(f'Nonlinearity \"{name}\" not defined.')

    return [lambda x: x] + [phi_l] * (n_layers - 1)


def create_cost(name, beta):
    """
    Create a supervised learning cost function used to nudge
    the network towards a desired state during training.

    Args:
        name: Name of the cost function
        beta: Scalar weighting factor of the cost function

    Returns:
        CEnergy object
    """
    if name == "squared_error":
        return cost.SquaredError(beta)
    elif name == "cross_entropy":
        return cost.CrossEntropy(beta)
    else:
        raise ValueError("Cost function \"{}\" not defined".format(name))


def create_optimizer(model, name, **kwargs):
    """
    Create optimizer for the given model.

    Args:
        model: nn.Module whose parameters will be optimized
        name: Name of the optimizer to be used

    Returns:
        torch.optim.Optimizer instance for the given model
    """
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))
