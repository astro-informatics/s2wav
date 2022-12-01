import pytest
import numpy as np
import pys2let as s2let

from s2wav import synthesis


L_to_test = [8, 16, 32]
N_to_test = [4, 6, 8]
J_min_to_test = [0]
lam_to_test = [2, 3]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_synthesis(wavelet_generator, L: int, N: int, J_min: int, lam: int):
    f_wav, f_scal = wavelet_generator(L=L, N=N, J_min=J_min, lam=lam)

    f = s2let.synthesis_wav2px(
        f_wav.flatten("C"),
        f_scal.flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_check = synthesis.synthesis_transform(f_wav, f_scal, L, N, J_min, lam)

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_vectorised_synthesis(wavelet_generator, L: int, N: int, J_min: int, lam: int):
    f_wav, f_scal = wavelet_generator(L=L, N=N, J_min=J_min, lam=lam)

    f = s2let.synthesis_wav2px(
        f_wav.flatten("C"),
        f_scal.flatten("C"),
        lam,
        L,
        J_min,
        N,
        0,
        upsample=True,
    )
    f_check = synthesis.vectorised_synthesis_transform(f_wav, f_scal, L, N, J_min, lam)

    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)
