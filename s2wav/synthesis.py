import numpy as np
from s2wav import samples, filters
import pyssht as ssht
import so3


def synthesis_transform(
    f_wav: np.ndarray, f_scal: np.ndarray, L: int, N: int, lam: float, J_min: int,
) -> np.ndarray:
    # TODO: call inbuilt functions to address shapes.
    assert f_wav.shape[0] == (J - J_min) * (2 * N - 1) * L * (2 * L - 1)
    assert f_scal.shape[0] == L * (2 * L - 1)

    flm = _synthesis_wav2lm(f_wav, f_scal, L, N, lam, J_min)

    return ssht.inverse(flm, L)


def _synthesis_wav2lm(
    f_wav: np.ndarray, f_scal: np.ndarray, L: int, N: int, lam: float, J_min: int,
) -> np.ndarray:
    J = samples.j_max(L, lam)
    # TODO: clean up interaction with ssht.
    f_scal_lm = ssht.forward(f_scal.reshape(L,2*L-1), L)
    f_wav_lmn = np.zeros((J - J_min) * (2 * N - 1) * L * L, dtype=np.complex128)

    # Offsets and
    offset = 0
    offset_lmn = 0
    inc = (2 * N - 1) * L * L
    inc_lmn = (2 * N - 1) * (2 * L - 1) * L

    for j in range(J_min, J + 1):
        # TODO: refactor kernel type "s2dw"
        L0 = samples.L0("s2dw", lam, j)
        f_wav_lmn[offset_lmn : offset_lmn + inc_lmn] = so3.forward(
            f_wav[offset : offset + inc], so3.create_parameter_dict(L=L, N=N)
        )
        offset += inc
        offset_lmn += inc_lmn

    return _synthesis_lmn2lmn(f_wav_lmn, f_scal_lm, L, N, lam, J_min)


def _synthesis_lmn2lmn(
    f_wav_lmn: np.ndarray, f_scal_lm: np.ndarray, L: int, N: int, lam: float, J_min: int,
) -> np.ndarray:
    # TODO: refactor spin and avoid so3.create_parameters.
    spin = original_spin = 0
    params = so3.create_parameter_dict(L=L, N=N)

    wav_lm, scal_l = filters.filters_directional(L, lam, spin, original_spin, N, J_min)

    flm = np.zeros(L**2, dtype=np.complex128)
    J = samples.j_max(L, lam)

    offset = 0
    inc = (2 * N - 1) * L * L

    for j in range(J_min, J + 1):
        for n in range(-N + 1, N, 2):
            for el in range(max(np.abs(spin), np.abs(n)), L):
                lm_ind = samples.elm2ind(el, n)
                psi = wav_lm[j * L * L + lm_ind]
                for m in range(-el, el + 1):
                    lm_ind = samples.elm2ind(el, m)
                    lmn_ind = samples.elmn2ind(el, m, n, L, N)
                    flm[lm_ind] += f_wav_lmn[offset + lmn_ind] * psi

        offset += inc

    for el in range(np.abs(spin), L):
        phi = np.sqrt(4 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            lm_ind = samples.elm2ind(el, m)
            flm[lm_ind] += f_scal_lm[lm_ind] * phi

    return flm


if __name__ == "__main__":
    L = 10
    N = 4
    J_min = 0
    lam = 2
    J = samples.j_max(L, lam)

    f_wav_size = (J - J_min) * (2 * N - 1) * L * (2 * L - 1)
    f_scal_size = L * (2 * L - 1)
    f_wav = np.random.randn(f_wav_size) + 1j*np.random.randn(f_wav_size)
    f_scal = np.random.randn(f_scal_size) + 1j*np.random.randn(f_scal_size)

    #f = synthesis_transform(f_wav, f_scal, L, N, lam, J_min)
    #f_true = s2let.synthesis(f_wav, f_scal)
