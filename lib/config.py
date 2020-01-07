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

import logging
import os
import time
from importlib import reload

import torch

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_dir = None
log_name = None
writer = None


def setup_logging(name, dir=""):
    """
    Setup the logging device to log into a uniquely created directory.

    Args:
        name: Name of the directory for the log-files.
        dir: Optional sub-directory within log
    """
    # Setup global log name and directory
    global log_name
    log_name = name

    # Setup global logging directory
    global log_dir
    log_dir = os.path.join("log", dir)

    # Create the logging folder if it does not exist already
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Need to reload logging as otherwise the logger might be captured by another library
    reload(logging)

    # Setup global logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-5.5s %(asctime)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(
                log_dir, time.strftime("%Y%m%d_%H%M") + "_" + name + ".log")
            ),
            logging.StreamHandler()
        ])
