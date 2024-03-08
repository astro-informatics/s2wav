import pytest
import numpy as np
import pys2let as s2let
from s2fft import base_transforms as sht_base
from s2wav.transforms import rec_wav_jax, pre_wav_jax, construct
from s2wav import filters, samples

L_to_test = [8]
N_to_test = [2, 3]
J_min_to_test = [2]
lam_to_test = [2, 3]
reality = [False, True]
sampling_to_test = ["mw", "mwss", "dh"]
recursive_transform = [False, True]

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
def test_jax_synthesis(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    recursive: bool
):
    J = samples.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
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
        upsample=False,
    )

    filter = filters.filters_directional_vectorised(L, N, J_min, lam)
    generator = construct.generate_wigner_precomputes if recursive else construct.generate_full_precomputes
    synthesis = rec_wav_jax.synthesis if recursive else pre_wav_jax.synthesis

    precomps = generator(
        L,
        N,
        J_min,
        lam,
        forward=True,
        reality=reality
    ) 
    f_check = synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        filters=filter,
        precomps=precomps
    )
    f = np.real(f) if reality else f
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
def test_jax_analysis(
    flm_generator,
    f_wav_converter,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    recursive: bool
):
    J = samples.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = sht_base.spherical.inverse(flm, L, reality=reality)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C").astype(np.complex128),
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=False
    )
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)
    generator = construct.generate_wigner_precomputes if recursive else construct.generate_full_precomputes
    analysis = rec_wav_jax.analysis if recursive else pre_wav_jax.analysis
    precomps = generator(
        L,
        N,
        J_min,
        lam,
        forward=False,
        reality=reality
    )
    f_wav_check, f_scal_check = analysis(
        f,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        filters=filter,
        precomps=precomps
    )

    f_wav_check = f_wav_converter(f_wav_check, L, N, J_min, lam, True)

    np.testing.assert_allclose(f_wav, f_wav_check, atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("sampling", sampling_to_test)
def test_jax_round_trip(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    sampling: str
):
    J = samples.j_max(L, lam)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = sht_base.spherical.inverse(flm, L, reality=reality, sampling=sampling)
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)

    f_wav, f_scal = rec_wav_jax.analysis(
        f,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        sampling=sampling,
        filters=filter
    )
    f_check = rec_wav_jax.synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        sampling=sampling,
        reality=reality,
        filters=filter
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
