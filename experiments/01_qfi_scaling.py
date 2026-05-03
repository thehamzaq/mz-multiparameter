"""
Experiment 01: Single-parameter QFI scaling for several probe states.

For SU(2) Mach-Zehnder metrology with the in-between state χ:
    F(θ) = 4 Var_χ(Jx)    (reflectivity QFI)
    F(φ) = 4 Var_χ(Jz)    (phase QFI)

Heisenberg limit: F = N². SQL: F = N.

This experiment scans probes in {sine, NOON, twin Fock, equator coherent} for
N ∈ [4, 60] and reports the asymptotic prefactor F/N².
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (sine_state, noon_state, twin_fock, coherent_spin,
                 qfi_reflectivity_analytic, qfi_phase_analytic)


def fit_loglog(Ns, F):
    Ns, F = np.asarray(Ns, float), np.asarray(F, float)
    m = (F > 1e-10) & np.isfinite(F)
    if m.sum() < 3:
        return float("nan")
    s, _ = np.polyfit(np.log(Ns[m]), np.log(F[m]), 1)
    return float(s)


def main():
    Ns = list(range(4, 61, 2))
    probes = [
        ("sine (Berry-Wiseman)", sine_state),
        ("NOON",                  noon_state),
        ("twin Fock",             lambda N: twin_fock(N) if N % 2 == 0 else None),
        ("equator coherent",      lambda N: coherent_spin(N, np.pi / 2, 0)),
    ]

    rows = []
    for name, builder in probes:
        F_th, F_ph = [], []
        Ns_ok = []
        for N in Ns:
            chi = builder(N)
            if chi is None:
                continue
            Ns_ok.append(N)
            F_th.append(qfi_reflectivity_analytic(N, chi))
            F_ph.append(qfi_phase_analytic(N, chi))
        slope_th = fit_loglog(Ns_ok, F_th)
        slope_ph = fit_loglog(Ns_ok, F_ph)
        const_th = F_th[-1] / Ns_ok[-1] ** 2
        const_ph = F_ph[-1] / Ns_ok[-1] ** 2
        rows.append({
            "probe": name,
            "F_theta_slope": slope_th,
            "F_phi_slope": slope_ph,
            "F_theta_over_N2_at_max": const_th,
            "F_phi_over_N2_at_max": const_ph,
            "Ns": Ns_ok,
            "F_theta": F_th,
            "F_phi": F_ph,
        })

    # Print summary
    print(f"{'probe':<24}{'slope F_θ':>12}{'slope F_φ':>12}"
          f"{'F_θ/N² (N=60)':>18}{'F_φ/N² (N=60)':>18}")
    print("-" * 84)
    for r in rows:
        print(f"{r['probe']:<24}{r['F_theta_slope']:>12.4f}{r['F_phi_slope']:>12.4f}"
              f"{r['F_theta_over_N2_at_max']:>18.4f}{r['F_phi_over_N2_at_max']:>18.4f}")
    print("\nSlope ≈ 2 ⇒ Heisenberg scaling. Slope ≈ 1 ⇒ SQL.")
    print("(Asymptotic prefactor of Heisenberg-saturating probes ≤ 1.)")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "01_qfi_scaling.json")
    with open(out_path, "w") as f:
        json.dump({"description": __doc__, "rows": rows}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
