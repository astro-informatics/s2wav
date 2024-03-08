:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2WAV`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2WAV`` package is structured such that the 2 primary transforms, the analysis and 
synthesis directional wavelet transforms, can easily be accessed.

Usage |:rocket:|
-----------------
To import and use ``S2WAV``  is as simple follows: 

.. code-block:: Python 

    import s2wav 

    # Compute wavelet coefficients
    f_wav, f_scal = s2wav.analysis(f, L, N)

    # Map back to signal on the sphere 
    f = s2wav.synthesis(f_wav, f_scal, L, N)

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   numpy_transform/numpy_transforms.nblink
   jax_transform/jax_transforms.nblink
   torch_transform/torch_transforms.nblink
