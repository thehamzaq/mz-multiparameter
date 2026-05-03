"""
Holevo Cramer-Rao Bound via SDP, ported from Albarelli-Friel-Datta 2019
(arXiv:1906.05724). Adapts the eigenbasis decomposition to a real-valued
SDP that cvxpy/SCS can solve. Fixes the npar=3 hardcode in the tantrix10 port.
"""
import numpy as np
import cvxpy as cp


def _rank(D, tol=1e-9):
    mask = D > tol
    return D[mask], int(mask.sum())


def _Smat(snonzero, d, rnk, fulldim):
    """Build the inner-product matrix S in the real eigenbasis (sparse)."""
    if rnk == 0:
        return np.zeros((fulldim, fulldim), dtype=complex)
    mask = np.triu(np.ones((rnk, rnk), dtype=bool), 1)
    scols = np.tile(np.real(snonzero)[None, :], (rnk, 1))
    srows = scols.T
    siplsj = scols + srows
    siminsj = -srows + scols

    # Diagonal entries
    n_diag = rnk
    n_offRank = (rnk * rnk - rnk) // 2  # upper triangle of real off-diag
    n_offKern = rnk * (d - rnk)
    diag_entries = np.concatenate([
        snonzero,                    # diagonal block
        siplsj[mask],                # off-diag real (rank-rank)
        siplsj[mask],                # off-diag imag (rank-rank)
        np.tile(snonzero, d - rnk),  # off-diag real (rank-kernel)
        np.tile(snonzero, d - rnk),  # off-diag imag (rank-kernel)
    ])

    Smat = np.diag(diag_entries.astype(complex))

    # Add anti-diagonal coupling between Re and Im parts:
    # for off-diag (rank-rank): coupling between blocks at offset rnk and rnk+n_offRank
    if n_offRank > 0:
        offdRank = 1j * np.diag(siminsj[mask])
        s = rnk
        Smat[s:s + n_offRank, s + n_offRank:s + 2 * n_offRank] = offdRank
        Smat[s + n_offRank:s + 2 * n_offRank, s:s + n_offRank] = -offdRank

    # for off-diag (rank-kernel): coupling between Re and Im
    if n_offKern > 0:
        offdKer = -1j * np.diag(np.tile(snonzero, d - rnk))
        s = rnk + 2 * n_offRank
        Smat[s + n_offKern:s + 2 * n_offKern, s:s + n_offKern] = offdKer
        Smat[s:s + n_offKern, s + n_offKern:s + 2 * n_offKern] = -offdKer

    return Smat


def _drho_to_real_basis(drho, V, rnk, d):
    """Express ∂ρ in the real eigenbasis of ρ. Returns vector of length fulldim."""
    fulldim = 2 * rnk * d - rnk * rnk
    # Eigen-frame matrix
    drho_eig = V.conj().T @ drho @ V

    # Diagonal block (rank-rank diagonal): real entries
    diag_block = np.real(np.diagonal(drho_eig)[:rnk])

    # Off-diag (rank-rank, upper triangle): real and imag separately
    n_offRank = (rnk * rnk - rnk) // 2
    if n_offRank > 0:
        mask = np.triu(np.ones((rnk, rnk), dtype=bool), 1)
        off_RR_real = np.real(drho_eig[:rnk, :rnk][mask])
        off_RR_imag = np.imag(drho_eig[:rnk, :rnk][mask])
    else:
        off_RR_real = np.array([])
        off_RR_imag = np.array([])

    # Off-diag (rank-kernel): drho_eig[:rnk, rnk:] flattened col-major
    if d > rnk:
        ak = drho_eig[:rnk, rnk:].T.flatten()  # (d-rnk)·rnk entries
        off_RK_real = np.real(ak)
        off_RK_imag = np.imag(ak)
    else:
        off_RK_real = np.array([])
        off_RK_imag = np.array([])

    return np.concatenate([diag_block, off_RR_real, off_RR_imag,
                            off_RK_real, off_RK_imag])


def hcrb_sdp(rho, drho_list, weight=None, solver="SCS", verbose=False):
    """Holevo CR bound via SDP, npar = len(drho_list).

    Returns (hcrb_value, status). hcrb_value is min Tr[W V] over the SDP.
    """
    rho = (rho + rho.conj().T) / 2
    drho_list = [(D + D.conj().T) / 2 for D in drho_list]
    d = rho.shape[0]
    npar = len(drho_list)
    if weight is None:
        weight = np.eye(npar)

    D, V = np.linalg.eigh(rho)
    D = np.real(D)
    # Sort descending
    idx = np.argsort(D)[::-1]
    D = D[idx]
    V = V[:, idx]

    snonzero, rnk = _rank(D)
    if rnk == 0:
        return float("inf"), "rank-0 state"
    if rnk == d:
        # full-rank: use simplified
        pass

    fulldim = 2 * rnk * d - rnk * rnk
    Smat = _Smat(snonzero, d, rnk, fulldim)
    Smat = (Smat.conj().T + Smat) / 2

    # Reduce to rank of S (kernel may exist):
    Sd, Sv = np.linalg.eigh(Smat)
    Sd_real = np.real(Sd)
    s_keep = Sd_real > 1e-9 * max(Sd_real.max(), 1e-30)
    Sd_pos = Sd_real[s_keep]
    Sv_pos = Sv[:, s_keep]
    R = np.diag(np.sqrt(Sd_pos)) @ Sv_pos.conj().T  # eff_dim × fulldim
    eff_dim = R.shape[0]

    # Identity-with-2s on off-diag entries (to convert real-basis IP to ⟨X|Y⟩):
    idd = np.diag(np.concatenate([
        np.ones(rnk),
        2 * np.ones(fulldim - rnk),
    ]))

    # ∂ρ in real basis as (fulldim × npar) matrix:
    drhomat = np.zeros((fulldim, npar), dtype=float)
    for k in range(npar):
        drhomat[:, k] = _drho_to_real_basis(drho_list[k], V, rnk, d)

    # CVXPY SDP — use complex Hermitian variables directly
    V_var = cp.Variable((npar, npar), symmetric=True)
    X_var = cp.Variable((fulldim, npar))

    # The PSD constraint is the complex Hermitian block matrix
    # [V       X^T R^†]
    # [R X       I    ] >> 0   (Hermitian PSD)
    # In cvxpy with complex matrices: bmat with complex blocks, then >> 0.
    R_complex = cp.Constant(R)
    block = cp.bmat([
        [V_var,                            X_var.T @ R_complex.H],
        [R_complex @ X_var,                np.eye(eff_dim)],
    ])

    constraints = [
        block >> 0,
        X_var.T @ idd @ drhomat == np.eye(npar),
    ]

    obj = cp.Minimize(cp.trace(weight @ V_var))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)
        return float(prob.value) if prob.value is not None else float("nan"), prob.status
    except Exception as e:
        return float("nan"), str(e)


if __name__ == "__main__":
    # Test: pure rotated twin Fock at η=1 should give HCRB = SLD bound = 4N/(N+2)
    from core import (angular_momentum, twin_fock, bs, phase_shift,
                      lossy_density_matrix, sld_qfi, _sld_qfi_pair)
    N = 4
    Jx, _, Jz = angular_momentum(N)
    chi = bs(np.pi / 2, Jx) @ twin_fock(N)
    psi_in = bs(-np.pi / 2, Jx) @ chi

    print("Test 1: η=1 pure state (HCRB should equal SLD = 4N/(N+2) = 8/6 ≈ 1.333)")
    eta = 1.0
    rho0 = lossy_density_matrix(N, psi_in, eta, np.pi / 2, np.pi / 2, np.pi / 2)
    h = 1e-4
    d_th = (lossy_density_matrix(N, psi_in, eta, np.pi / 2 + h, np.pi / 2, np.pi / 2)
            - lossy_density_matrix(N, psi_in, eta, np.pi / 2 - h, np.pi / 2, np.pi / 2)) / (2 * h)
    d_ph = (lossy_density_matrix(N, psi_in, eta, np.pi / 2, np.pi / 2, np.pi / 2 + h)
            - lossy_density_matrix(N, psi_in, eta, np.pi / 2, np.pi / 2, np.pi / 2 - h)) / (2 * h)
    F_th = sld_qfi(rho0, d_th)
    F_ph = sld_qfi(rho0, d_ph)
    F_thph = _sld_qfi_pair(rho0, d_th, d_ph)
    F = np.array([[F_th, F_thph], [F_thph, F_ph]])
    sld = float(np.trace(np.linalg.inv(F)))
    val, status = hcrb_sdp(rho0, [d_th, d_ph])
    bound = 4 / (N * (N + 2))
    print(f"  N={N}, op-pt symmetric, η={eta}:")
    print(f"    SLD CRB    = {sld:.6f}  (× N² = {sld*N**2:.4f})")
    print(f"    Holevo SDP = {val:.6f}  (× N² = {val*N**2 if val == val else float('nan'):.4f})")
    print(f"    Theoretical bound 4/(N(N+2)) = {bound:.6f}")
    print(f"    SDP status: {status}")
