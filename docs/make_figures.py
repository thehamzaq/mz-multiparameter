"""Generate figures for the README. Reads from results/*.json."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fig_qfi_scaling():
    data = load("01_qfi_scaling.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    colors = {"sine (Berry-Wiseman)": "#1f77b4", "NOON": "#d62728",
              "twin Fock": "#2ca02c", "equator coherent": "#ff7f0e"}
    for r in data["rows"]:
        Ns = np.array(r["Ns"])
        F_th = np.array(r["F_theta"])
        F_ph = np.array(r["F_phi"])
        c = colors.get(r["probe"], "gray")
        axes[0].loglog(Ns, F_th, "o-", label=r["probe"], color=c)
        axes[1].loglog(Ns, F_ph, "o-", label=r["probe"], color=c)
    Ns_ref = np.array([4, 60])
    for ax in axes:
        ax.loglog(Ns_ref, Ns_ref, "k--", alpha=0.4, label="SQL: N")
        ax.loglog(Ns_ref, Ns_ref ** 2, "k:", alpha=0.4, label="HL: N²")
        ax.set_xlabel("N (photons)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("$F_\\theta$  (reflectivity QFI)")
    axes[0].set_title("Reflectivity QFI vs N")
    axes[1].set_ylabel("$F_\\varphi$  (phase QFI)")
    axes[1].set_title("Phase QFI vs N")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "fig_qfi_scaling.png"), dpi=140)
    plt.close()
    print("Wrote docs/fig_qfi_scaling.png")


def fig_joint_bound():
    data = load("02_joint_bound.json")
    Ns = [r["N"] for r in data["rows"]]
    achieved = [r["N2_Tr_Finv"] for r in data["rows"]]
    bound = [r["bound_4N_over_Np2"] for r in data["rows"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ns, achieved, "o-", label="rotated twin Fock (achieved)", color="#2ca02c")
    ax.plot(Ns, bound, "k--", label="$4N/(N+2)$ bound", alpha=0.7)
    ax.axhline(4, color="r", linestyle=":", alpha=0.5, label="asymptotic limit = 4")
    ax.set_xlabel("N (photons)")
    ax.set_ylabel("$N^2 \\cdot \\mathrm{Tr}[F^{-1}]$")
    ax.set_title("Joint Cramér-Rao bound saturation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "fig_joint_bound.png"), dpi=140)
    plt.close()
    print("Wrote docs/fig_joint_bound.png")


def fig_invariances():
    data = load("05_invariances.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Dephasing panel: F_θθ flat, F_φφ collapses
    deph = data["dephasing"]
    Ns = sorted({r["N"] for r in deph})
    for N in Ns:
        rows = [r for r in deph if r["N"] == N]
        gammas = [r["gamma"] for r in rows]
        F_th = [r["F_theta_theta"] for r in rows]
        F_ph = [r["F_phi_phi"] for r in rows]
        axes[0].plot(gammas, np.array(F_th) / (N * (N + 2) / 2), "o-",
                     label=f"$F_\\theta$, N={N}")
        axes[0].plot(gammas, np.array(F_ph) / (N * (N + 2) / 2), "x:",
                     label=f"$F_\\varphi$, N={N}", alpha=0.7)
    axes[0].set_xlabel("dephasing strength $\\gamma$")
    axes[0].set_ylabel("QFI / [N(N+2)/2]")
    axes[0].set_title("$F_\\theta$ exactly invariant under Jz-dephasing")
    axes[0].set_xscale("symlog", linthresh=1e-3)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1, color="k", linestyle="--", alpha=0.3)

    # Asymmetric loss panel
    asym = data["asymmetric_loss"]
    Ns = sorted({r["N"] for r in asym})
    for N in Ns:
        rows = [r for r in asym if r["N"] == N]
        etas = [r["eta_a"] for r in rows]
        ratios = [r["ratio"] for r in rows]
        axes[1].plot(etas, ratios, "o-", label=f"N={N}")
    axes[1].set_xlabel("$\\eta_a$ (mode-a survival, with $\\eta_b = 1$)")
    axes[1].set_ylabel("$F_\\theta(\\eta_a, 1) / F_\\theta(1, 1)$")
    axes[1].set_title("$F_\\theta$ exactly preserved under one-arm loss")
    axes[1].set_ylim(0.99, 1.01)
    axes[1].axhline(1, color="k", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "fig_invariances.png"), dpi=140)
    plt.close()
    print("Wrote docs/fig_invariances.png")


def fig_loss_sweep():
    data = load("06_loss_sweep.json")
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns_plot = [16]
    colors = {"rot Twin Fock": "#2ca02c", "sine (BW)": "#1f77b4", "NOON": "#d62728"}
    for N in Ns_plot:
        for probe in ["rot Twin Fock", "sine (BW)", "NOON"]:
            sub = [r for r in rows if r["N"] == N and r["probe"] == probe]
            if not sub:
                continue
            etas = [r["eta"] for r in sub]
            tis = [r["N2_Tr_Finv"] if r["N2_Tr_Finv"] is not None else np.nan
                   for r in sub]
            ax.semilogy(etas, tis, "o-", label=f"{probe} (N={N})",
                         color=colors.get(probe, "gray"))
    bound_at_N = 4 * 16 / (16 + 2)
    ax.axhline(bound_at_N, color="k", linestyle="--", alpha=0.5,
                label=f"joint CR bound (N=16) = {bound_at_N:.3f}")
    ax.set_xlabel("transmission $\\eta$ (symmetric)")
    ax.set_ylabel("$N^2 \\cdot \\mathrm{Tr}[F^{-1}]$  (lower = better)")
    ax.set_title("Joint estimation under symmetric photon loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "fig_loss_sweep.png"), dpi=140)
    plt.close()
    print("Wrote docs/fig_loss_sweep.png")


if __name__ == "__main__":
    fig_qfi_scaling()
    fig_joint_bound()
    fig_invariances()
    fig_loss_sweep()
    print("\nAll figures generated.")
