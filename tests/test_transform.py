import pytest
import numpy as np
import pys2let as s2let

from s2wav import synthesis, analysis


L_to_test = [8, 10]
N_to_test = [1, 2, 3]
J_min_to_test = [0, 1]
lam_to_test = [2, 3]
multiresolution = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
def test_synthesis_looped(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
):
    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=lam, multiresolution=multiresolution
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
def test_synthesis_vectorised(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    multiresolution: bool,
):
    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=lam, multiresolution=multiresolution
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
        f_wav, f_scal, L, N, J_min, lam, multiresolution=multiresolution
    )

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_analysis(L: int, N: int, J_min: int, lam: int):
    f = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_wav_check, f_scal_check = analysis.analysis_transform(f, L, N, J_min, lam)

    np.testing.assert_allclose(f_wav, f_wav_check.flatten("C"), atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_analysis_vectorised(L: int, N: int, J_min: int, lam: int):
    f = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_wav_check, f_scal_check = analysis.analysis_transform_vectorised(
        f, L, N, J_min, lam
    )

    np.testing.assert_allclose(f_wav, f_wav_check.flatten("C"), atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)
