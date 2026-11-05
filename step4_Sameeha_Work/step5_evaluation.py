"""
step5_evaluation.py -> Task 2 (Sameeha): checking every estimator against the true MuJoCo values

What's in here:
    estimate_all()   runs every estimator from step 4 on one trial
    metrics()        angle and speed error (RMSE) per joint + the biggest speed error
    nees()           NEES = checks if the Kalman filter is honest about its own uncertainty

Uses info from tutorials: Week 7: estimate vs truth plots, +-2 sigma bounds and NEES.
Criterion: C (estimation results)

Run:  python step4_Sameeha_Work/step5_evaluation.py
      -> graphs in results/ and results/step5_accuracy_table.csv (for the project page)
"""

import sys
# csv is imported to save the results table for the project page
import csv
from pathlib import Path

# numpy is imported for the error maths
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing what I need from steps 1, 3 and 4
from step4_Sameeha_Work.step1_state_space_model import (                 # noqa: E402
    load_scene, linearise, double_integrator, NAMES, RESULTS)
from step4_Sameeha_Work.step3_measurement_and_ground_truth import run_trial, truth, Rn_for   # noqa: E402
from step4_Sameeha_Work.step4_observers import (                         # noqa: E402
    finite_difference, luenberger_gain, run_luenberger, run_kf, rmse, W_O, Q_A)

K0_S = 1.0          # s, I leave out the first second (start-up) from all the error numbers

# colours for each method so they're the same in every graph
STYLE = {"finite difference": "0.65", "model only (L = 0)": "C1", "const.-velocity KF": "C4",
         "Luenberger": "C2", "KF (MuJoCo model)": "C3"}


def estimate_all(tr, lin, w_o=W_O, q_a=Q_A, Rn=None):
    """lin = (Ad, Bd, x_star, u_star), the normal model. Returns {name: estimates} and the KF's P."""
    # This is to run all 5 estimators on one trial so I can compare them
    Ad, Bd, xs, us = lin
    dt = tr["t"][1] - tr["t"][0]
    n = tr["y"].shape[1]
    Ad_di, Bd_di, C = double_integrator(dt, n)
    # Luenberger gain with my tuned W_O
    L = luenberger_gain(Ad, C, w_o, dt)
    # MAIN PART: my Kalman filter, also keeping P (its uncertainty) for the +-2 sigma and NEES checks
    X_kf, P_kf = run_kf(tr, q_a, Ad, Bd, xs, us, Rn=Rn)
    # dictionary of every method: name -> its estimate
    est = {"finite difference": finite_difference(tr),
           "model only (L = 0)": run_luenberger(tr, np.zeros_like(L), Ad, Bd, xs, us),
           # the no-model KF: double integrator with u set to 0
           "const.-velocity KF": run_kf(dict(tr, u=np.zeros_like(tr["u"])), 2.0e3, Ad_di, Bd_di,
                                        np.zeros(2 * n), np.zeros(n), Rn=Rn)[0],
           "Luenberger": run_luenberger(tr, L, Ad, Bd, xs, us),
           "KF (MuJoCo model)": X_kf}
    return est, P_kf


def metrics(X_hat, X, k0):
    # This is to calculate the error = estimate - truth (skipping the first k0 steps of start-up)
    E = X_hat[k0:] - X[k0:]
    # gives angle RMSE, speed RMSE and the biggest speed error, for each joint
    return rmse(E[:, :6]), rmse(E[:, 6:]), np.abs(E[:, 6:]).max(0)


def nees(X_hat, P, X, k0):
    # NEES = error' P^-1 error. If the KF is honest about its uncertainty this averages about 12 (number of states)
    E = X_hat[k0:] - X[k0:]
    # einsum does that calculation for every time step at once
    return np.einsum("ki,kij,kj->k", E, np.linalg.inv(P[k0:]), E)


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # scipy stats is imported for the chi-squared numbers (the NEES pass band)
    from scipy import stats

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    # same setup as step 4: scene, model at home, and the 10 s test trial with seed 1
    model, arm, dt, q_home = load_scene()
    lin = linearise(arm, q_home)
    tr = run_trial(arm, q_home, T=10.0, seed=1)
    X, t = truth(tr), tr["t"]
    k0 = int(K0_S / dt)
    # MAIN PART: running every estimator on the trial
    est, P_kf = estimate_all(tr, lin)

    # ------------------------------------------------------------------ C.1 speed estimates
    # top rows: true speed vs the KF (dashed on top). Bottom rows: the speed ERROR of each method,
    # because the good ones sit right on top of the truth and finite difference is too noisy to show.
    # only plotting 3-6 s so you can actually see the lines
    w = slice(int(3 / dt), int(6 / dt))
    fd_rmse = metrics(est["finite difference"], X, k0)[1].mean()
    # 4 rows x 3 columns: speed with its error underneath, for the 6 joints
    fig, ax = plt.subplots(4, 3, figsize=(13, 9), sharex=True)
    for j in range(6):
        # picking which subplot this joint's speed and error go in
        a_v = ax[2 * (j // 3), j % 3]
        a_e = ax[2 * (j // 3) + 1, j % 3]
        # top: true speed (black) and my KF (red dashed)
        a_v.plot(t[w], X[w, 6 + j], "k", lw=2.5, label="MuJoCo truth")
        a_v.plot(t[w], est["KF (MuJoCo model)"][w, 6 + j], "C3--", lw=1.2, label="KF estimate")
        a_v.set_title(NAMES[j]); a_v.set_ylabel("qd [rad/s]")
        # bottom: the error of each method (should be close to 0)
        for lab in ["const.-velocity KF", "Luenberger", "KF (MuJoCo model)"]:
            a_e.plot(t[w], est[lab][w, 6 + j] - X[w, 6 + j], color=STYLE[lab], lw=0.8, label=lab)
        a_e.set_ylabel("error [rad/s]")
        a_e.set_ylim(-0.25, 0.25)
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=6, title=f"finite difference: RMSE {fd_rmse:.2f} rad/s (off the scale)",
                    title_fontsize=6)
    for a in ax[3]:
        a.set_xlabel("t [s]")
    fig.suptitle("C.1 estimated joint speeds (not measured) vs MuJoCo truth, 3-6 s")
    plt.tight_layout(); fig.savefig(RESULTS / "step5_velocity_estimates.png", dpi=150)

    # ------------------------------------------------------------------ C.2 error inside +-2 sigma
    # the KF's error at every step
    E_kf = est["KF (MuJoCo model)"] - X
    # sigma = square root of the diagonal of P = how unsure the KF says it is about each state
    sig = np.sqrt(np.einsum("kii->ki", P_kf))
    fig, ax = plt.subplots(2, 6, figsize=(15, 4.8), sharex=True)
    for j in range(6):
        for row, off, unit in [(0, 0, "rad"), (1, 6, "rad/s")]:
            a = ax[row, j]
            # red = actual error, blue band = +-2 sigma. The red should stay inside the blue ~95 % of the time
            a.plot(t, E_kf[:, off + j], "C3", lw=0.6)
            a.fill_between(t, -2 * sig[:, off + j], 2 * sig[:, off + j], color="C0", alpha=0.25)
            # zooming the y-axis so the band fills the graph
            lim = 4 * np.median(sig[k0:, off + j])
            a.set_ylim(-lim, lim)
            if j == 0:
                a.set_ylabel(("angle" if row == 0 else "speed") + f" error [{unit}]")
            if row == 0:
                a.set_title(NAMES[j])
            else:
                a.set_xlabel("t [s]")
    fig.suptitle("C.2 KF error (red) should stay inside its own +-2 sigma band from P (blue)")
    plt.tight_layout(); fig.savefig(RESULTS / "step5_kf_error_bounds.png", dpi=150)
    # % of errors inside the +-2 sigma band (should be about 95 %)
    inside = 100 * np.mean(np.abs(E_kf[k0:]) < 2 * sig[k0:], axis=0)

    # ------------------------------------------------------------------ C.3 results table
    print("=" * 86)
    print("C.3 - errors vs MuJoCo truth (normal trial, 1-10 s)")
    print("=" * 86)
    print(f"{'method':22s} {'angle RMSE [mrad]':>17s} {'speed RMSE [rad/s]':>19s} {'max speed err':>14s}")
    rows = []
    # working out the errors for each method, printing them and keeping them for the CSV
    for lab, Xh in est.items():
        rp, rv, mv = metrics(Xh, X, k0)
        rows.append([lab, *np.round(1e3 * rp, 4), *np.round(rv, 5)])
        print(f"{lab:22s} {1e3 * rp.mean():17.3f} {rv.mean():19.4f} {mv.max():14.3f}")
    print("raw encoder noise (to compare):", f"{1e3 * np.sqrt(Rn_for()[0, 0]):.3f} mrad")
    # saving the table as a CSV for the project page
    with open(RESULTS / "step5_accuracy_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["estimator"] + [f"pos_rmse_mrad_{n}" for n in NAMES] + [f"vel_rmse_rads_{n}" for n in NAMES])
        wr.writerows(rows)

    # bar graph of the errors per joint for each method -> step5_accuracy_bars.png
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    wb = 0.16
    for i, (lab, Xh) in enumerate(est.items()):
        rp, rv, _ = metrics(Xh, X, k0)
        ax[0].bar(np.arange(6) + i * wb, 1e3 * rp, wb, color=STYLE[lab], label=lab)
        ax[1].bar(np.arange(6) + i * wb, rv, wb, color=STYLE[lab], label=lab)
    for a, yl in zip(ax, ["angle RMSE [mrad]", "speed RMSE [rad/s]"]):
        a.set_xticks(np.arange(6) + 2 * wb); a.set_xticklabels(NAMES); a.set_ylabel(yl)
    ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
    fig.suptitle("C.3 errors per joint (normal trial)")
    plt.tight_layout(); fig.savefig(RESULTS / "step5_accuracy_bars.png", dpi=150)

    # ------------------------------------------------------------------ C.4 NEES
    # NEES of my KF over the trial
    eps = nees(est["KF (MuJoCo model)"], P_kf, X, k0)
    # the 95 % band the AVERAGE NEES should be inside if the KF is honest
    lo, hi = stats.chi2.ppf([0.025, 0.975], len(eps) * 12) / len(eps)
    print("\n" + "=" * 86)
    print("C.2/C.4 - is the KF honest about how unsure it is?")
    print("=" * 86)
    print("% of errors inside +-2 sigma   q :", np.round(inside[:6], 1))
    print("                               qd:", np.round(inside[6:], 1), "  (should be ~95 %)")
    print(f"mean NEES = {eps.mean():.2f}   95% band for the mean = [{lo:.2f}, {hi:.2f}]   (n = 12)")
    print("  above the band = too confident (Qd too small);  below = too cautious (Qd too big)")

    # graph of NEES over time with the n = 12 line -> step5_nees.png
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.semilogy(t[k0:], eps, "C3", lw=0.5, label="NEES")
    ax.axhline(12, color="k", ls="--", label="n = 12")
    ax.axhline(stats.chi2.ppf(0.975, 12), color="0.5", ls=":", label="97.5 % for one sample")
    ax.set_xlabel("t [s]"); ax.set_ylabel("NEES"); ax.legend(fontsize=8)
    ax.set_title("C.4 KF consistency (normal trial)")
    plt.tight_layout(); fig.savefig(RESULTS / "step5_nees.png", dpi=150)

    print("""
What these results mean
  * In the normal trial my linear model is almost perfect and u is known, so even 'model only'
    (no encoder correction at all) looks really good. The normal case makes any model look good.
    The encoders matter when the real robot is different from the model (push, payload, other
    poses) - that's step 7, and it's why step 6 tunes on a mix of situations.
  * Q_A was tuned in step 6 on that mix, so on this easy trial the KF is a bit too cautious on
    purpose (NEES below 12, errors well inside +-2 sigma). It's set up for the hard cases.
  * Finite difference is ~100x worse than any observer at dt = 2 ms (noise x sqrt(2)/dt).
  * The constant-velocity KF has no robot model, so the gap between it and my KF shows what
    the MuJoCo model from step 1 actually adds.
""")
    print("figures + table saved to", RESULTS)
    plt.show()