import pytest
import numpy as np
import pys2let as s2let

from s2wav import synthesis, analysis, samples
import s2fft


L_to_test = [8, 10]
N_to_test = [1, 2, 3]
J_min_to_test = [0, 1]
lam_to_test = [2, 3]
multiresolution = [False, True]
reality = [False, True]
sampling_to_test = ['mw', 'mwss', 'dh']


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_synthesis_looped(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
):
    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=lam, multiresolution=multiresolution, reality=reality
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

    f_check = synthesis.synthesis_transform_looped(
        f_wav, f_scal, L, N, J_min, lam, multiresolution=multiresolution
    )

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_synthesis_vectorised(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
):
    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=lam, multiresolution=multiresolution, reality=reality
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

    f_check = synthesis.synthesis_transform_vectorised(
        f_wav, f_scal, L, N, J_min, lam, multiresolution=multiresolution, reality=reality
    )
    f = np.real(f) if reality else f
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_analysis_looped(
    flm_generator,
    f_wav_converter,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
):
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.transform.inverse(flm, L, reality=reality)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C").astype(np.complex128),
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=not multiresolution,
    )
    f_wav_check, f_scal_check = analysis.analysis_transform_looped(
        f, L, N, J_min, lam, reality=reality, multiresolution=multiresolution
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
def test_analysis_vectorised(
    flm_generator,
    f_wav_converter,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
):
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.transform.inverse(flm, L, reality=reality)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C").astype(np.complex128),
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=not multiresolution,
    )
    f_wav_check, f_scal_check = analysis.analysis_transform_vectorised(
        f, L, N, J_min, lam, multiresolution=multiresolution, reality=reality
    )

    f_wav_check = f_wav_converter(
        f_wav_check, L, N, J_min, lam, multiresolution
    )
    np.testing.assert_allclose(f_wav, f_wav_check.flatten("C"), atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("sampling", sampling_to_test)
def test_looped_round_trip(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
    sampling: str,
):

    nside = int(L/2)

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.transform.inverse(flm, L, reality=reality, sampling=sampling, nside=nside)

    f_wav, f_scal = analysis.analysis_transform_looped(
        f, L, N, J_min, lam, multiresolution=multiresolution, reality=reality, sampling=sampling, nside=nside
    )
    print(f_wav[0].shape, f_scal.shape)

    f_check = synthesis.synthesis_transform_looped(
        f_wav, f_scal, L, N, J_min, lam, multiresolution=multiresolution, sampling=sampling, nside=nside
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("sampling", sampling_to_test)
def test_vectorised_round_trip(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
    reality: bool,
    sampling: str,
):

    nside = int(L/2)

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.transform.inverse(flm, L, reality=reality, sampling=sampling, nside=nside)

    f_wav, f_scal = analysis.analysis_transform_vectorised(
        f, L, N, J_min, lam, multiresolution=multiresolution, reality=reality, sampling=sampling, nside=nside
    )
    print(f_wav[0].shape, f_scal.shape)

    f_check = synthesis.synthesis_transform_vectorised(
        f_wav, f_scal, L, N, J_min, lam, multiresolution=multiresolution, sampling=sampling, nside=nside
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)