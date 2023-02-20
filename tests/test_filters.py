import pytest
import numpy as np
from s2wav.filter_factory import filters, tiling
from s2wav.utils.shapes import j_max

L_to_test = [8, 16, 32]
N_to_test = [4, 6, 8]
J_min_to_test = [0]
lam_to_test = [2, 3]

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_axisym_admissibility(L: int, J_min: int, lam: int):
    Psi, Phi = filters.filters_axisym(L, J_min, lam)
    J = j_max(L, lam)
    Psi_j_sum = np.zeros_like(Phi)
    for j in range(J_min, J + 1):
        for el in range(L):
            Psi_j_sum[el] += np.abs(Psi[j, el]) ** 2

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
    spin = 0
    psi, phi = filters.filters_directional(L, N, J_min, lam)
    J = j_max(L, lam)

    ident = np.zeros(L, dtype=np.complex128)

    for el in range(np.abs(spin), L):
        ident[el] += 4.0 * np.pi / (2 * el + 1) * phi[el] * phi[el]

    for j in range(J + 1):
        for el in range(np.abs(spin), L):
            for m in range(-el, el + 1):
                ident[el] += (
                    8.0
                    * np.pi
                    * np.pi
                    / (2 * el + 1)
                    * psi[j, el, L - 1 + m]
                    * np.conj(psi[j, el, L - 1 + m])
                )

    for el in range(max(np.abs(spin), 1), L):
        assert ident[el] == pytest.approx(
            1 + 0j, rel=1e-14
        ), "Admissibility condition not satisfied at l = " + str(el)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
def test_directional_tiling(L: int, N: int):
    s_elm = tiling.tiling_direction(L, N)
    for el in range(1, L):
        temp = 0
        for m in range(-el, el + 1):
            temp += s_elm[el, L - 1 + m] * np.conj(s_elm[el, L - 1 + m])

        assert temp == pytest.approx(
            1 + 0j, rel=1e-14
        ), "Directional tiling satisfied l = " + str(el)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_axisym_vectorised(L: int, J_min: int, lam: int):
    f = filters.filters_axisym(L, J_min, lam)
    f_vect = filters.filters_axisym_vectorised(L, J_min, lam)
    for i in range(2):
        np.testing.assert_allclose(f[i], f_vect[i], rtol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_directional_vectorised(L: int, N: int, J_min: int, lam: int):
    f = filters.filters_directional(L, N, J_min, lam)
    f_vect = filters.filters_directional_vectorised(L, N, J_min, lam)

    for i in range(2):
        np.testing.assert_allclose(f[i], f_vect[i], rtol=1e-14)



@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_axisym_jax(L: int, J_min: int, lam: int):
    f = filters.filters_axisym(L, J_min, lam)
    f_jax = np.array(filters.filters_axisym_jax(L, J_min, lam))
    for i in range(2):
        np.testing.assert_allclose(f[i], f_jax[i], rtol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_directional_jax(L: int, N: int, J_min: int, lam: int):
    f = filters.filters_directional(L, N, J_min, lam)
    f_jax = np.array(filters.filters_directional_jax(L, N, J_min, lam))

    for i in range(2):
        np.testing.assert_allclose(f[i], f_jax[i], rtol=1e-14)
