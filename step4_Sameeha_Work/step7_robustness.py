"""
step7_robustness.py -> Task 2 (Sameeha): does the estimate still work when things go wrong?

The tests:
    E.1  push           a 10 N m push on the shoulder lift that the observer doesn't know about
    E.2  payload        0 to 1.5 kg added to the wrist (like a picked-up object), observer keeps
                        the normal model
    E.3  random poses   20 trials with the start pose moved by up to +-0.5 rad per joint (the
                        object moves between trials, so the pose changes too - the brief asks for this)
    E.4  extra          re-linearising the model as the arm moves, on a motion twice as big

Each test compares: finite difference, model only (L = 0), Luenberger and my KF.

Uses info from tutorials: Week 8 Part B: testing under noise and disturbance, what separation
doesn't guarantee and Week 7 C.2: re-linearising around the current estimate (like the EKF).
Criterion: E (integration and robustness)

Run:  python step4_Sameeha_Work/step7_robustness.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from step4_Sameeha_Work.step1_state_space_model import load_scene, linearise, double_integrator, NAMES, RESULTS  # noqa
from step4_Sameeha_Work.step3_measurement_and_ground_truth import run_trial, truth, AMP   # noqa: E402
from step4_Sameeha_Work.step4_observers import (                         # noqa: E402
    finite_difference, luenberger_gain, run_luenberger, run_kf, rmse, W_O, Q_A)
from step4_Sameeha_Work.step5_evaluation import metrics, nees, K0_S, STYLE   # noqa: E402
from step4_Sameeha_Work.step6_tuning import payload_model, push, PUSH_T, PUSH_LEN  # noqa: E402

LABELS = ["finite difference", "model only (L = 0)", "Luenberger", "KF (MuJoCo model)"]


def estimates(tr, lin, L):
    Ad, Bd, xs, us = lin
    return {"finite difference": finite_difference(tr),
            "model only (L = 0)": run_luenberger(tr, np.zeros_like(L), Ad, Bd, xs, us),
            "Luenberger": run_luenberger(tr, L, Ad, Bd, xs, us),
            "KF (MuJoCo model)": run_kf(tr, Q_A, Ad, Bd, xs, us)[0]}


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    model, arm, dt, q_home = load_scene()
    lin = linearise(arm, q_home)
    _, _, C = double_integrator(dt)
    L = luenberger_gain(lin[0], C, W_O, dt)
    k0 = int(K0_S / dt)

    # ------------------------------------------------------------------ E.1 push
    tr = run_trial(arm, q_home, T=8.0, seed=31, disturbance=push)
    X, t = truth(tr), tr["t"]
    est = estimates(tr, lin, L)
    w = slice(int((PUSH_T - 0.5) / dt), int((PUSH_T + 1.5) / dt))
    after = slice(int(PUSH_T / dt), int((PUSH_T + 1.0) / dt))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for lab in LABELS:
        if lab != "finite difference":
            ax[0].plot(t[w], 1e3 * (est[lab][w, 1] - X[w, 1]), color=STYLE[lab], label=lab)
            ax[1].plot(t[w], est[lab][w, 7] - X[w, 7], color=STYLE[lab], label=lab)
    ax[2].plot(t[w], X[w, 7], "k", lw=2, label="truth")
    ax[2].plot(t[w], est["KF (MuJoCo model)"][w, 7], "C3", label="KF")
    for a in ax:
        a.axvspan(PUSH_T, PUSH_T + PUSH_LEN, color="C1", alpha=0.15); a.set_xlabel("t [s]")
    ax[0].set_ylabel("lift angle error [mrad]"); ax[1].set_ylabel("lift speed error [rad/s]")
    ax[2].set_ylabel("lift speed [rad/s]"); ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.suptitle("E.1 a 10 N m push on the shoulder lift that the observer doesn't know about (shaded)")
    plt.tight_layout(); fig.savefig(RESULTS / "step7_push.png", dpi=150)

    print("=" * 82)
    print("E.1 - push: shoulder lift error in the 1 s after the push starts")
    print("=" * 82)
    for lab in LABELS:
        e = est[lab][after] - X[after]
        print(f"  {lab:20s} angle RMSE {1e3 * rmse(e[:, 1]):7.3f} mrad   speed RMSE {rmse(e[:, 7]):.4f} rad/s"
              f"   biggest speed error {np.abs(e[:, 7]).max():.4f}")

    # ------------------------------------------------------------------ E.2 payload
    payloads = [0.0, 0.25, 0.5, 1.0, 1.5]
    pr = {k: np.zeros((len(payloads), 2)) for k in LABELS}
    for i, kg in enumerate(payloads):
        tr = run_trial(arm, q_home, T=8.0, seed=41, model=payload_model(kg))
        X = truth(tr)
        for lab, Xh in estimates(tr, lin, L).items():
            rp, rv, _ = metrics(Xh, X, k0)
            pr[lab][i] = 1e3 * rp.mean(), rv.mean()
    print("\n" + "=" * 82)
    print("E.2 - weight on the wrist; the observers still use the NORMAL model")
    print("=" * 82)
    print("  payload [kg]  " + "".join(f"{lab:>26s}" for lab in LABELS))
    print("                " + "".join(f"{'angle [mrad] | speed [rad/s]':>26s}" for _ in LABELS))
    for i, kg in enumerate(payloads):
        print(f"  {kg:10.2f}    " + "".join(f"{pr[lab][i, 0]:13.3f} | {pr[lab][i, 1]:8.4f}" for lab in LABELS))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for lab in LABELS[1:]:
        ax[0].plot(payloads, pr[lab][:, 0], "o-", color=STYLE[lab], label=lab)
        ax[1].plot(payloads, pr[lab][:, 1], "o-", color=STYLE[lab], label=lab)
    ax[0].axhline(1.0, color="k", ls="-.", lw=1, label="raw encoder noise")
    ax[0].set_ylabel("angle RMSE [mrad]"); ax[1].set_ylabel("speed RMSE [rad/s]")
    for a in ax:
        a.set_xlabel("payload [kg]"); a.legend(fontsize=7)
    fig.suptitle("E.2 a picked-up object the model doesn't know about")
    plt.tight_layout(); fig.savefig(RESULTS / "step7_payload.png", dpi=150)

    # ------------------------------------------------------------------ E.3 random poses
    N_MC = 20
    rng = np.random.default_rng(7)
    mc_p = np.zeros((N_MC, len(LABELS))); mc_v = np.zeros_like(mc_p)
    for i in range(N_MC):
        q0 = q_home + rng.uniform(-0.5, 0.5, 6)
        kg = rng.uniform(0.0, 1.0)                       # plus a random unknown payload each time
        tr = run_trial(arm, q0, T=6.0, seed=100 + i, model=payload_model(kg))
        X = truth(tr)
        for j, (lab, Xh) in enumerate(estimates(tr, lin, L).items()):
            rp, rv, _ = metrics(Xh, X, k0)
            mc_p[i, j], mc_v[i, j] = 1e3 * rp.mean(), rv.mean()
    print("\n" + "=" * 82)
    print(f"E.3 - {N_MC} trials: start pose +-0.5 rad per joint AND payload 0-1 kg, normal model at home")
    print("=" * 82)
    for j, lab in enumerate(LABELS):
        print(f"  {lab:20s} angle [mrad] median {np.median(mc_p[:, j]):7.3f} worst {mc_p[:, j].max():7.3f}"
              f"    speed [rad/s] median {np.median(mc_v[:, j]):.4f} worst {mc_v[:, j].max():.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].boxplot(mc_p, tick_labels=[l.replace(" (", "\n(") for l in LABELS])
    ax[1].boxplot(mc_v, tick_labels=[l.replace(" (", "\n(") for l in LABELS])
    ax[0].axhline(1.0, color="k", ls="-.", lw=1)
    ax[0].set_ylabel("angle RMSE [mrad]"); ax[1].set_ylabel("speed RMSE [rad/s]")
    for a in ax:
        a.set_yscale("log"); a.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"E.3 {N_MC} random trials (pose and payload), one normal model")
    plt.tight_layout(); fig.savefig(RESULTS / "step7_monte_carlo.png", dpi=150)
    np.savetxt(RESULTS / "step7_monte_carlo.csv", np.hstack([mc_p, mc_v]), delimiter=",", fmt="%.6f",
               header=",".join([f"pos_mrad_{l}" for l in LABELS] + [f"vel_rads_{l}" for l in LABELS]))

    # ------------------------------------------------------------------ E.4 re-linearised KF
    tr = run_trial(arm, q_home, T=8.0, seed=51, amp=2.0 * AMP, model=payload_model(0.5))
    X = truth(tr)
    Ad, Bd, xs, us = lin
    X_fix, P_fix = run_kf(tr, Q_A, Ad, Bd, xs, us)
    X_rel, P_rel = run_kf(tr, Q_A, Ad, Bd, xs, us, arm=arm, relin_every=int(0.2 / dt))
    print("\n" + "=" * 82)
    print("E.4 - extra: re-linearise the model every 0.2 s (motion 2x bigger, 0.5 kg payload)")
    print("=" * 82)
    for lab, Xh, Ph in [("fixed model at home", X_fix, P_fix), ("re-linearised", X_rel, P_rel)]:
        rp, rv, _ = metrics(Xh, X, k0)
        print(f"  {lab:22s} angle RMSE {1e3 * rp.mean():.3f} mrad   speed RMSE {rv.mean():.4f} rad/s"
              f"   mean NEES {nees(Xh, Ph, X, k0).mean():.1f}")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(tr["t"], 1e3 * np.abs(X_fix[:, :6] - X[:, :6]).mean(1), "C3", lw=0.7, label="fixed model")
    ax.plot(tr["t"], 1e3 * np.abs(X_rel[:, :6] - X[:, :6]).mean(1), "C4", lw=0.7, label="re-linearised")
    ax.set_xlabel("t [s]"); ax.set_ylabel("average |angle error| [mrad]"); ax.legend(fontsize=8)
    ax.set_title("E.4 big motion (2x) with a 0.5 kg payload")
    plt.tight_layout(); fig.savefig(RESULTS / "step7_relinearised_kf.png", dpi=150)

    print("""
What these results mean
  * 'model only' gets the speed right but can't tell when there's a load. With a payload (E.2, E.3)
    its angle error grows with the weight and never goes away, and during the push (E.1) it's wrong
    until the servo recovers. The encoder correction fixes both - that's the whole point of the observer.
  * The KF and Luenberger keep the angle error well below the raw encoder noise in every test,
    and the KF is the most accurate overall (E.2, E.3).
  * During the push, Luenberger is better than the KF on angle. The KF assumes small random
    accelerations, but a push is a sudden jump, and the faster Luenberger reacts quicker.
    This is a limit of my Qd choice; a fix would be adding a disturbance-torque state to x.
  * E.4: re-linearising didn't help. The stiff servos (step 1) control the joints and hardly change
    with pose, so one model at home is enough. That's still a useful result: it shows the simpler
    fixed model in observer.py is fine.
  * Separation principle (Week 8): designing the observer and controller separately is only
    guaranteed if the model is right. E.2 and E.3 are exactly the "model is wrong" cases, which is
    why I tested them instead of just assuming it works.
""")

    print("figures saved to", RESULTS)
    plt.show()
