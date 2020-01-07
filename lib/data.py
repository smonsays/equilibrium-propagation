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
from torchvision import datasets, transforms


def _one_hot_ten(label):
    """
    Helper function to convert to a one hot encoding with 10 classes.

    Args:
        label: target label as single number

    Returns:
        One-hot tensor with dimension (*, 10) encoding label
    """
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)


def create_mnist_loaders(batch_size):
    """
    Create dataloaders for the training and test set of MNIST.

    Args:
        batch_size: Number of samples per batch

    Returns:
        train_loader: torch.utils.data.DataLoader for the MNIST training set
        test_loader: torch.utils.data.DataLoader for the MNIST test set
    """

    # Load train and test MNIST datasets
    mnist_train = datasets.MNIST('../data/', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                 ]),
                                 target_transform=_one_hot_ten
                                 )

    mnist_test = datasets.MNIST('../data/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ]),
                                target_transform=_one_hot_ten
                                )

    # For GPU acceleration store dataloader in pinned (page-locked) memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create the dataloader objects
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)

    return train_loader, test_loader
