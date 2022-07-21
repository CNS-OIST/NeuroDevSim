NeuroDevSim Installation
************************

NeuroDevSim only runs in Linux or MacOS (including ARM64) environments, but unfortunately not in Windows environments. This is because Windows does not support multiprocess forking.

**Mimimum Prerequisites**

1. Python 3.6 or more recent (tested on Python 3.8.3 and 3.9.7)
2. `NumPy <http://www.numpy.org/>`_
3. `Scipy <http://www.scipy.org/>`_
4. `Matplotlib <http://matplotlib.org/>`_
5. `colorama <https://pypi.python.org/pypi/colorama>`_

If you are new to this, an easy way to get all software at once is to install `Anaconda <http://www.anaconda.com/>`_. On ARM64 based Macs install Miniforge3 as described in first part of `Tensorflow-plugin <https://developer.apple.com/metal/tensorflow-plugin/>`_, afterwards use `conda install` to install the libraries listed above.

To run NeuroDevSim models you need a `jupyter notebook <http://jupyter.org/>`_ and/or a `terminal <https://en.wikipedia.org/wiki/List_of_terminal_emulators>`_ application. To make movies you will need to instal the `FFmpeg <https://www.ffmpeg.org/download.html>`_ tool.

**Install From Github Repository**

To install NeuroDevSim from source code, first clone the repository using the following commands in terminal::

    cd your_location
    git clone https://github.com/CNS-OIST/NeuroDevSim.git
    pip install your_location/neurodevsim

