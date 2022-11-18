import pytest
import numpy as np 
from s2wav import axisym_filters
from s2wav import helper_functions

L_to_test = [8, 16, 32]
J_min_to_test = [1,2]
lam_to_test = [2]

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
def test_axisym_admissibility(L: int, J_min: int, lam: int):
    """Test the calculation of filters in the axisymmetric case by verifying the admissibility condition given 
    by equation (9) in [1] stands.
    
    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    print("L {}, J_min {}, lam {}".format(L, J_min, lam))
    Psi, Phi = axisym_filters.filters_axisym(L, lam, J_min) 
    J = helper_functions.j_max(L, lam)
    Psi_j_sum = np.zeros_like(Phi)
    for j in range(J_min, J+1):
        for l in range(L):
            Psi_j_sum[l] += np.abs(Psi[l + j * L])**2
        
    for l in range(L):
         norm = 1
         temp = norm * (Phi[l]**2 + Psi_j_sum[l])
         assert temp == pytest.approx(1, rel=1e-14), "Admissibility condition not satisfied at l = "+str(l)
        