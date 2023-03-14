[![image](https://img.shields.io/badge/GitHub-s2wav-blue.svg?style=flat)](https://github.com/astro-informatics/s2wav)
[![image](https://github.com/astro-informatics/s2wav/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2wav/actions/workflows/tests.yml)
[![image](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://astro-informatics.github.io/s2wav)
[![image](https://codecov.io/gh/astro-informatics/s2wav/branch/main/graph/badge.svg?token=ZES6J4K3KZ)](https://codecov.io/gh/astro-informatics/s2wav)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat)](https://arxiv.org/abs/xxxx.xxxxx)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img align="left" height="85" width="98" src="./docs/assets/sax_logo.png">

# Differentiable and accelerated spherical wavelets with JAX

`S2WAV` is a JAX package for computing wavelet transforms on the sphere
and rotation group. It leverages autodiff to provide differentiable
transforms, which are also deployable on modern hardware accelerators
(e.g. GPUs and TPUs), and can be mapped across multiple accelerators.

More specifically, `S2WAV` provides support for scale-discretised
wavelet transforms on the sphere and rotation group (for both real and
complex signals), with support for adjoints where needed, and comes with
a variety of different optimisations (e.g. precompute or not,
multi-resolution algorithms) that one may select depending on available
resources and desired angular resolution $L$.

## Installation :computer:

The Python dependencies for the `S2WAV` package are listed in the file
`requirements/requirements-core.txt` and will be automatically installed
into the active python environment by [pip]{https://pypi.org} when running

``` bash
pip install .        
```

from the root directory of the repository. Unit tests can then be
executed to ensure the installation was successful by running

``` bash
pytest tests/         # for pytest
```

In the near future one will be able to install `S2WAV` directly from
[PyPi]{https://pypi.org} by `pip install s2wav` but this is not yet supported.
Note that to run `JAX` on NVIDIA GPUs you will need to follow the
[guide](https://github.com/google/jax#installation) outlined by Google.

## Usage :rocket:

To import and use `S2WAV` is as simple follows:

``` python
import s2wav 

# Compute wavelet coefficients
f_wav, f_scal = s2wav.analysis(f, L, N)

# Map back to signal on the sphere 
f = s2wav.synthesis(f_wav, f_scal, L, N)
```

## Contributors ✨

TODO: Add core contributors photos etc here pre-release.

We strongly encourage contributions from any interested developers; a
simple example would be adding support for more spherical sampling
patterns!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Attribution

A BibTeX entry for `S2WAV` is:

``` 
@article{price:s2wav, 
   AUTHOR = {Author names},
    TITLE = {"TBA"},
   EPRINT = {arXiv:0000.00000},
     YEAR = {2023}
}
```

## License :memo:

Copyright 2023 Matthew Price, Jessica Whtiney, Alicja Polanska, Jason
McEwen and contributors.

`S2WAV` is free software made available under the MIT License. For
details see the LICENSE file.
