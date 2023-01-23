import pytest
import pys2let as s2let
import numpy as np


from s2wav import analysis, synthesis

L_to_test = [8, 16, 32]
N_to_test = [4, 6, 8]
J_min_to_test = [0]
lam_to_test = [2, 3]

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_reality_synthesis(wavelet_generator, L: int, N: int, J_min: int, lam: int):

    f_wav, f_scal = (wavelet_generator(L=L, N=N, J_min=J_min, lam=lam))


    f = s2let.synthesis_wav2px(
        (f_wav.real+0j).flatten("C"),
        (f_scal.real+0j).flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_check = synthesis.synthesis_transform(f_wav.real, f_scal.real, L, N, J_min, lam, reality=True)

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_reality_analysis(wavelet_generator, L: int, N: int, J_min: int, lam: int):
    f = np.random.randn(L, 2 * L - 1) + 0j
    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_wav_check, f_scal_check = analysis.analysis_transform(f.real, L, N, J_min, lam, reality=True)

    np.testing.assert_allclose(f_wav, f_wav_check.flatten("C"), atol=1e-14)
    np.testing.assert_allclose(f_scal, f_scal_check.flatten("C"), atol=1e-14)    