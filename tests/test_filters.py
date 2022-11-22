import pytest
import numpy as np
from s2wav import filters, samples, tiling

L_to_test = [8, 16, 32]
N_to_test = [4, 6, 8]
J_min_to_test = [0]
lam_to_test = [2, 3]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_axisym_admissibility(L: int, J_min: int, lam: int):
    Psi, Phi = filters.filters_axisym(L, lam, J_min)
    J = samples.j_max(L, lam)
    Psi_j_sum = np.zeros_like(Phi)
    for j in range(J_min, J + 1):
        for el in range(L):
            Psi_j_sum[el] += np.abs(Psi[el + j * L]) ** 2

    for el in range(L):
        temp = Phi[el] ** 2 + Psi_j_sum[el]
        assert temp == pytest.approx(
            1, rel=1e-14
        ), "Admissibility condition not satisfied at l = " + str(el)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_directional_admissibility(L: int, N: int, J_min: int, lam: int):
    original_spin = spin = 0

    psi, phi = filters.filters_directional(L, lam, spin, original_spin, N, J_min)
    J = samples.j_max(L, lam)

    ident = np.zeros(L, dtype=np.complex128)

    for el in range(np.abs(spin), L):
        ident[el] += 4.0 * np.pi / (2 * el + 1) * phi[el] * phi[el]

    for j in range(J + 1):
        ind = spin * spin
        for el in range(np.abs(spin), L):
            for m in range(-el, el + 1):
                ident[el] += (
                    8.0
                    * np.pi
                    * np.pi
                    / (2 * el + 1)
                    * psi[j * L * L + ind]
                    * np.conj(psi[j * L * L + ind])
                )
                ind += 1

    for el in range(max(np.abs(spin), 1), L):
        assert ident[el] == pytest.approx(
            1 + 0j, rel=1e-14
        ), "Admissibility condition not satisfied at l = " + str(el)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_directional_tiling(L: int, N: int):
    s_elm = tiling.tiling_direction(N, L)
    ind = 1
    for el in range(1, L):
        temp = 0
        for m in range(-el, el + 1):
            temp += s_elm[ind] * np.conj(s_elm[ind])
            ind += 1

        assert temp == pytest.approx(
            1 + 0j, rel=1e-14
        ), "Directional tiling satisfied l = " + str(el)
