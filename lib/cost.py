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

import abc

import torch


class CEnergy(abc.ABC):
    """
    Abstract base class for all supervised learning cost functions.

    Attributes:
        beta: Scalar weighting factor of the cost function
        target: Current target tensor for the cost function
    """
    def __init__(self, beta):
        super(CEnergy, self).__init__()
        self.beta = beta
        self.target = None

    @abc.abstractmethod
    def compute_energy(self, u_last):
        """
        Compute energy/loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        return

    def set_target(self, target):
        """
        Set new target tensor for the cost function.

        Args:
            target: target tensor
        """
        self.target = target


class CrossEntropy(CEnergy):
    """
    Cross entropy cost function.
    """
    def __init__(self, beta):
        super(CrossEntropy, self).__init__(beta)

    def compute_energy(self, u_last):
        """
        Compute cross-entropy loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        loss = torch.nn.functional.cross_entropy(u_last, self.target, reduction='none')
        return self.beta * loss

    def set_target(self, target):
        if target is None:
            self.target = None
        else:
            # Need to transform target for the F.cross_entropy function
            self.target = target.argmax(dim=1)


class SquaredError(CEnergy):
    """
    Squared energy cost function.
    """
    def __init__(self, beta):
        super(SquaredError, self).__init__(beta)

    def compute_energy(self, u_last):
        """
        Compute mean squared error loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        loss = torch.nn.functional.mse_loss(u_last, self.target.float(), reduction='none')
        return self.beta * 0.5 * torch.sum(loss, dim=1)
