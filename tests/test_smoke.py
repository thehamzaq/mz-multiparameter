"""Smoke tests for the toolkit. Run via `python -m pytest tests/`."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (angular_momentum, twin_fock, bs,
                 qfi_two_param_inbetween, qfi_pure)


def rotated_twin_fock(N):
    Jx, _, _ = angular_momentum(N)
    return bs(np.pi / 2, Jx) @ twin_fock(N)


def test_angular_momentum_commutator():
    """[Jx, Jy] = i Jz on the symmetric subspace."""
    for N in [4, 6, 8]:
        Jx, Jy, Jz = angular_momentum(N)
        comm = Jx @ Jy - Jy @ Jx
        target = 1j * Jz
        assert np.allclose(comm, target, atol=1e-10), \
            f"[Jx, Jy] != i Jz at N={N}"


def test_jy_anticommutator_identity():
    """Jy|j,0⟩ = -i {Jx, Jz}|j,0⟩ on twin Fock for integer j."""
    for N in [4, 6, 8, 10, 12]:
        Jx, Jy, Jz = angular_momentum(N)
        tf = twin_fock(N)
        lhs = Jy @ tf
        rhs = -1j * (Jx @ Jz + Jz @ Jx) @ tf
        assert np.allclose(lhs, rhs, atol=1e-12), \
            f"Identity violated at N={N}"


def test_rotated_twin_fock_saturates_bound():
    """Rotated twin Fock saturates Tr[F⁻¹] = 4/(N(N+2))."""
    for N in [4, 6, 8, 10, 14]:
        chi = rotated_twin_fock(N)
        F = qfi_two_param_inbetween(N, chi)
        det = F[0, 0] * F[1, 1] - F[0, 1] ** 2
        trinv = (F[0, 0] + F[1, 1]) / det
        bound = 4 / (N * (N + 2))
        assert abs(trinv - bound) / bound < 1e-10, \
            f"N={N}: Tr[F⁻¹]={trinv}, bound={bound}, gap={trinv - bound}"


def test_compatibility_zero_for_rotated_twin_fock():
    """⟨Jy⟩ = 0 on the rotated twin Fock probe (Matsumoto compatibility)."""
    from src.core import expval
    for N in [4, 6, 8]:
        Jx, Jy, Jz = angular_momentum(N)
        chi = rotated_twin_fock(N)
        ey = expval(Jy, chi)
        assert abs(ey) < 1e-12, f"N={N}: ⟨Jy⟩ = {ey}"


if __name__ == "__main__":
    # Run smoke tests directly without pytest
    test_angular_momentum_commutator()
    print("PASS  test_angular_momentum_commutator")
    test_jy_anticommutator_identity()
    print("PASS  test_jy_anticommutator_identity")
    test_rotated_twin_fock_saturates_bound()
    print("PASS  test_rotated_twin_fock_saturates_bound")
    test_compatibility_zero_for_rotated_twin_fock()
    print("PASS  test_compatibility_zero_for_rotated_twin_fock")
    print("\nAll smoke tests pass.")
