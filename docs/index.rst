Differentiable and accelerated spherical wavelets
===================================================

``S2WAV`` is a JAX package for computing wavelet transforms on the sphere and rotation 
group.  It leverages autodiff to provide differentiable transforms, which are also 
deployable on modern hardware accelerators (e.g. GPUs and TPUs), and can be mapped 
across multiple accelerators.

More specifically, ``S2WAV`` provides support for scale-discretised wavelet transforms 
on the sphere and rotation group (for both real and complex signals), with support for 
adjoints where needed, and comes with a variety of different optimisations (e.g. precompute 
or not, multi-resolution algorithms) that one may select depending on available resources 
and desired angular resolution :math:`L`.

Contributors |:hammer:|
------------------------
TODO: Add core contributors photos etc here pre-release.

We strongly encourage contributions from any interested developers; a simple example would be adding 
support for more spherical sampling patterns!

Attribution |:books:|
----------------------
A BibTeX entry for ``S2WAV`` is:

.. code-block:: 

     @article{price:s2wav, 
        AUTHOR = {Author names},
         TITLE = {"TBA"},
        EPRINT = {arXiv:0000.00000},
          YEAR = {2023}
     }

License |:memo:|
-----------------

Copyright 2023 Matthew Price, Jessica Whtiney, Alicja Polanska, Jason McEwen and contributors.

``S2WAV`` is free software made available under the MIT License. For details see
the LICENSE file.

.. bibliography:: 
    :notcited:
    :list: bullet

.. * :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Interactive Tutorials
   
   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   api/index

