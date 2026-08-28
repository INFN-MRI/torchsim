TorchSim
========

TorchSim is a pure Pytorch-based MR simulator, including analytical and EPG model.

|Coverage| |CI/CD| |License| |Codefactor| |Sphinx| |PyPi| |Black| |PythonVersion|

.. |Coverage| image:: https://codecov.io/gh/INFN-MRI/torchsim/graph/badge.svg?token=qtB53xANwI 
   :target: https://codecov.io/gh/INFN-MRI/torchsim

.. |CI/CD| image:: https://github.com/INFN-MRI/torchsim/workflows/CI-CD/badge.svg
   :target: https://github.com/INFN-MRI/torchsim

.. |License| image:: https://img.shields.io/github/license/INFN-MRI/torchsim
   :target: https://github.com/INFN-MRI/torchsim/blob/main/LICENSE.txt

.. |Codefactor| image:: https://www.codefactor.io/repository/github/INFN-MRI/torchsim/badge
   :target: https://www.codefactor.io/repository/github/INFN-MRI/torchsim

.. |Sphinx| image:: https://img.shields.io/badge/docs-Sphinx-blue
   :target: https://infn-mri.github.io/torchsim

.. |PyPi| image:: https://img.shields.io/pypi/v/torchsim
   :target: https://pypi.org/project/torchsim

.. |Black| image:: https://img.shields.io/badge/style-black-black

.. |PythonVersion| image:: https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white
   :target: https://python.org

Features
--------
TorchSim contains tools to implement parallelized and differentiable MR simulators. Specifically, we provide

1. Automatic vectorization of across multiple atoms (e.g., voxels).
2. Automatic generation of forward and jacobian methods (based on forward-mode autodiff) to be used in parameter fitting or model-based reconstructions.
3. Support for custom manual defined jacobian methods to override auto-generated jacobian.
4. Support for advanced signal models, including diffusion, flow, magnetization transfer and chemical exchange.
5. GPU support.

Installation
------------

TorchSim can be installed via pip as:

.. code-block:: bash

    pip install torchsim

Basic Usage
-----------
Using TorchSim, we can quickly implement and run MR simulations.
We also provide pre-defined simulators for several applications:

.. code-block:: python
    
    import numpy as np
    import torchsim
    
    # generate a flip angle pattern
    flip = np.concatenate((np.linspace(5, 60.0, 300), np.linspace(60.0, 2.0, 300), np.ones(280)*2.0))
    sig, jac = torchsim.mrf_sim(flip=flip, TR=10.0, T1=1000.0, T2=100.0, diff=("T1","T2"))
    
This way we obtained the forward pass signal (``sig``) as well as the jacobian
calculated with respect to ``T1`` and ``T2``.


Development
-----------

If you are interested in improving this project, install TorchSim in editable mode:

.. code-block:: bash

    git clone git@github.com:INFN-MRI/torchsim
    cd torchsim
    pip install -e .[dev]


Related projects
----------------

This package is inspired by the following excellent projects:

- epyg <https://github.com/brennerd11/EpyG>
- sycomore <https://github.com/lamyj/sycomore/>
- mri-sim-py <https://somnathrakshit.github.io/projects/project-mri-sim-py-epg/>
- ssfp <https://github.com/mckib2/ssfp>
- erwin <https://github.com/lamyj/erwin>

