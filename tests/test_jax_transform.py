from jax import jit, config

config.update("jax_enable_x64", True)

import pytest
import numpy as np
import pys2let as s2let

from s2wav.transforms import synthesis_jax, analysis
from s2wav.filter_factory import filters
from s2fft import base_transforms as base


L_to_test = [8, 10]
N_to_test = [1, 2, 3]
J_min_to_test = [0, 1]
lam_to_test = [2, 3]
multiresolution = [False, True]
reality = [False, True]
sampling_to_test = ["mw", "mwss", "dh"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_synthesis_jax(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
):
    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
        multiresolution=multiresolution,
        reality=reality,
    )

    f = s2let.synthesis_wav2px(
        f_wav_s2let,
        f_scal_s2let,
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=not multiresolution,
    )

    filter = filters.filters_directional_vectorised(L, N, J_min, lam)

    f_check = synthesis_jax.synthesis_transform_jax(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter
    )
    f = np.real(f) if reality else f
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)
