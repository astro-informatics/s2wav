:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2WAV`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2WAV`` package is structured such that the 2 primary transforms, the Wigner and 
spherical harmonic transforms, can easily be accessed.

Usage |:rocket:|
-----------------
To import and use ``S2WAV``  is as simple follows: 

.. code-block:: Python 

    import s2wav 

    # Compute wavelet coefficients
    f_wav, f_scal = s2wav.analysis(f, L, N)

    # Map back to signal on the sphere 
    f = s2wav.synthesis(f_wav, f_scal, L, N)

Benchmarking |:hourglass_flowing_sand:|
-------------------------------------
We benchmarked the spherical harmonic and Wigner transforms implemented in ``S2WAV``
against the C implementations in the `S2LET <https://github.com/astro-informatics/s2let>`_
pacakge. 

TODO: Add table here when results available 

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   transforms/wavelet_transform.nblink
   precompute_transforms/wavelet_transform_precompute.nblink
