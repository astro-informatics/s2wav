import pytest
import numpy as np
import pys2let as s2let

from s2wav.transforms import jax_wavelets
from s2wav.filter_factory import filters
from s2wav.utils import shapes
from s2fft import base_transforms as base

L_to_test = [6, 8]
N_to_test = [2, 3]
J_min_to_test = [1, 2]
lam_to_test = [2, 3]
multiresolution = [False, True]
reality = [False, True]
multiple_gpus = [False]
sampling_to_test = ["mw", "mwss", "dh"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("spmd", multiple_gpus)
def test_jax_synthesis(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
    spmd: bool,
):
    J = shapes.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

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

    # Precompute some values
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)
    precomps = jax_wavelets.generate_wigner_precomputes(
        L,
        N,
        J_min,
        lam,
        forward=True,
        reality=reality,
        multiresolution=multiresolution,
    )
    f_check = jax_wavelets.synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter,
        precomps=precomps,
        spmd=spmd,
    )
    f = np.real(f) if reality else f
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("spmd", multiple_gpus)
def test_jax_analysis(
    flm_generator,
    f_wav_converter,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
    spmd: bool,
):
    J = shapes.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = base.spherical.inverse(flm, L, reality=reality)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C").astype(np.complex128),
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=not multiresolution,
    )
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)
    precomps = jax_wavelets.generate_wigner_precomputes(
        L,
        N,
        J_min,
        lam,
        forward=False,
        reality=reality,
        multiresolution=multiresolution,
    )
    f_wav_check, f_scal_check = jax_wavelets.analysis(
        f,
        L,
        N,
        J_min,
        lam,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter,
        precomps=precomps,
        spmd=spmd,
    )

    f_wav_check = f_wav_converter(
        f_wav_check, L, N, J_min, lam, multiresolution
    )

    np.testing.assert_allclose(f_wav, f_wav_check, atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("spmd", multiple_gpus)
def test_jax_round_trip(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
    sampling: str,
    spmd: bool,
):
    J = shapes.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = base.spherical.inverse(flm, L, reality=reality, sampling=sampling)
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)

    f_wav, f_scal = jax_wavelets.analysis(
        f,
        L,
        N,
        J_min,
        lam,
        multiresolution=multiresolution,
        reality=reality,
        sampling=sampling,
        filters=filter,
        spmd=spmd,
    )
    f_check = jax_wavelets.synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        multiresolution=multiresolution,
        sampling=sampling,
        reality=reality,
        filters=filter,
        spmd=spmd,
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
