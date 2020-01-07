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

from lib import config


class EnergyBasedModel(abc.ABC, torch.nn.Module):
    """
    Abstract base class for all energy-based models.

    Attributes:
        batch_size: Number of samples per batch
        c_energy: Cost function to nudge last layer
        clamp_du: List of boolean values tracking if the corresponding layer
            is clamped to a fixed value
        E: Current energy of the model. Object has the responsibility of
            maintaining the energy consistent if u or W change.
        dimensions: Dimensions of the underlying multilayer perceptron
        n_layers: Number of layers of the multilayer perceptron
        phi: List of activation functions for each layer
        W: ModuleList for the linear layers of the multilayer perceptron
        u: List of pre-activations

    """
    def __init__(self, dimensions, c_energy, batch_size, phi):
        super(EnergyBasedModel, self).__init__()

        self.batch_size = batch_size
        self.c_energy = c_energy
        self.clamp_du = torch.zeros(len(dimensions), dtype=torch.bool)
        self.dimensions = dimensions
        self.E = None
        self.n_layers = len(dimensions)
        self.phi = phi
        self.u = None
        self.W = torch.nn.ModuleList(
            torch.nn.Linear(dim1, dim2)
            for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
        ).to(config.device)

        # Input (u_0) is clamped by default
        self.clamp_du[0] = True

        self.reset_state()

    @abc.abstractmethod
    def fast_init(self):
        """
        Fast initilization of pre-activations.
        """
        return

    @abc.abstractmethod
    def update_energy(self):
        """
        Update the energy.
        """
        return

    def clamp_layer(self, i, u_i):
        """
        Clamp the specified layer.

        Args:
            i: Index of layer to be clamped
            u_i: Tensor to which the layer i is clamped
        """
        self.u[i] = u_i
        self.clamp_du[i] = True
        self.update_energy()

    def release_layer(self, i):
        """
        Release (i.e. un-clamp) the specified layer).

        Args:
            i: Index of layer to be released
        """
        self.u[i].requires_grad = True
        self.clamp_du[i] = False
        self.update_energy()

    def reset_state(self):
        """
        Reset the state of the system to a random (Normal) configuration.
        """
        self.u = []
        for i in range(self.n_layers):
            self.u.append(torch.randn((self.batch_size, self.dimensions[i]),
                                      requires_grad=not(self.clamp_du[i]),
                                      device=config.device))
        self.update_energy()

    def set_C_target(self, target):
        """
        Set new target tensor for the cost function.

        Args:
            target: target tensor
        """
        self.c_energy.set_target(target)
        self.update_energy()

    def u_relax(self, dt, n_relax, tol, tau):
        """
        Relax the neural state variables until a fixed point is obtained
        with precision < tol or until the maximum number of steps n_relax is reached.

        Args:
            dt: Step size
            n_relax: Maximum number of steps
            tol: Tolerance/precision of relaxation
            tau: Time constant

        Returns:
            Change in energy after relaxation
        """
        E_init = self.E.clone().detach()
        E_prev = self.E.clone().detach()

        for i in range(n_relax):
            # Perform a single relaxation step
            du_norm = self.u_step(dt, tau)
            # dE = self.E.detach() - E_prev

            # If change is below numerical tolerance, break
            if du_norm < tol:
                break

            E_prev = self.E.clone().detach()

        return torch.sum(E_prev - E_init)

    def u_step(self, dt, tau):
        """
        Perform single relaxation step on the neural state variables.

        Args:
            dt: Step size
            tau: Time constant

        Returns:
            Absolute change in pre-activations
        """
        # Compute gradients wrt current energy
        self.zero_grad()
        batch_E = torch.sum(self.E)
        batch_E.backward()

        with torch.no_grad():
            # Apply the update in every layer
            du_norm = 0
            for i in range(self.n_layers):
                if not self.clamp_du[i]:
                    du = self.u[i].grad
                    self.u[i] -= dt / tau * du

                    du_norm += float(torch.mean(torch.norm(du, dim=1)))

        self.update_energy()

        return du_norm

    def w_get_gradients(self, loss=None):
        """
        Compute the gradient on the energy w.r.t. the parameters W.

        Args:
            loss: Optional loss to optimze for. Otherwise the mean energy is optimized.

        Returns:
            List of gradients for each layer
        """
        self.zero_grad()
        if loss is None:
            loss = torch.mean(self.E)
        return torch.autograd.grad(loss, self.parameters())

    def w_optimize(self, free_grad, nudged_grad, w_optimizer):
        """
        Update weights using free and nudged phase gradients.

        Args:
            free_grad: List of free phase gradients
            nudged_grad: List of nudged phase gradients
            w_optimizer: torch.optim.Optimizer for the model parameters
        """
        self.zero_grad()
        w_optimizer.zero_grad()

        # Apply the contrastive Hebbian style update
        for p, f_g, n_g in zip(self.parameters(), free_grad, nudged_grad):
            p.grad = (1 / self.c_energy.beta) * (n_g - f_g)

        w_optimizer.step()
        self.update_energy()

    def zero_grad(self):
        """
        Zero gradients for parameters and pre-activations.
        """
        self.W.zero_grad()
        for u_i in self.u:
            if u_i.grad is not None:
                u_i.grad.detach_()
                u_i.grad.zero_()


class ConditionalGaussian(EnergyBasedModel):
    """
    One example of an energy-based model that has a probabilistic interpretation as
    the (negative) log joint probability of a conditional-Gaussian model.
    Also see review by Bogacz and Whittington, 2019.
    """
    def __init__(self, dimensions, c_energy, batch_size, phi):
        super(ConditionalGaussian, self).__init__(dimensions, c_energy, batch_size, phi)

    def fast_init(self):
        """
        The FF init is a very handy hack when working with the ConditionalGaussian
        model, which allows reducing the number of fixed point iterations
        significantly, and results in improved training for large dt steps.
        """
        for i in range(self.n_layers - 1):
            self.u[i + 1] = self.W[i](self.phi[i](self.u[i])).detach()
            self.u[i + 1].requires_grad = not self.clamp_du[i + 1]

        self.update_energy()

    def update_energy(self):
        """
        Update the energy as the mean squared predictive error.
        """
        self.E = 0
        for i in range(self.n_layers - 1):
            pred = self.W[i](self.phi[i](self.u[i]))
            loss = torch.nn.functional.mse_loss(pred, self.u[i + 1], reduction='none')
            self.E += torch.sum(loss, dim=1)

        if self.c_energy.target is not None:
            self.E += self.c_energy.compute_energy(self.u[-1])


class RestrictedHopfield(EnergyBasedModel):
    """
    The classical Hopfield energy in a restricted feedforward model
    as used in the original equilibrium propagation paper by Scellier, 2017
    """
    def __init__(self, dimensions, c_energy, batch_size, phi):
        super(RestrictedHopfield, self).__init__(dimensions, c_energy, batch_size, phi)

    def fast_init(self):
        raise NotImplementedError("Fast initialization not possible for the Hopfield model.")

    def update_energy(self):
        """
        Update the energy computed as the Hopfield Energy.
        """
        self.E = 0

        for i, layer in enumerate(self.W):
            r_pre = self.phi[i](self.u[i])
            r_post = self.phi[i + 1](self.u[i + 1])

            if i == 0:
                self.E += 0.5 * torch.einsum('ij,ij->i', self.u[i], self.u[i])

            self.E += 0.5 * torch.einsum('ij,ij->i', self.u[i + 1], self.u[i + 1])
            self.E -= 0.5 * torch.einsum('bi,ji,bj->b', r_pre, layer.weight, r_post)
            self.E -= 0.5 * torch.einsum('bi,ij,bj->b', r_post, layer.weight, r_pre)
            self.E -= torch.einsum('i,ji->j', layer.bias, r_post)

        if self.c_energy.target is not None:
            self.E += self.c_energy.compute_energy(self.u[-1])
