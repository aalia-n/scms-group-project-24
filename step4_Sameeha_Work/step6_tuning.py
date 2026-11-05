"""
step6_tuning.py -> Task 2 (Sameeha): picking the observer settings (and why)

The trade-off: trust the model too much -> wrong when the real robot is different;
               trust the encoder too much -> noisy.

What's in here:
    payload_model()    the scene with extra weight on the wrist (like a picked-up object)
    push()             a short torque push the observer doesn't know about
    sustained_load()   a torque that turns on and stays on (object weight, cable, contact)
    tuning_set()       the 4 test situations: normal, other pose, payload, steady load.
                       If I only tuned on the normal one it would pick "just trust the model",
                       which goes wrong as soon as anything changes (see step 5).
    score()            one number for how good an estimate is (lower = better):
                         J = angle error / encoder noise + speed error / finite-difference noise
                       so it's how much of the raw sensor's error is left.

Uses info from tutorials: Week 7: slow vs fast observer, choosing Qd vs Rn and Week 6:
trade-off curves (only the Q/R ratio matters).
Criterion: C (tuning)

Run:  python step4_Sameeha_Work/step6_tuning.py
      -> prints the best W_O and Q_A. Copy them into step4_observers.py and observer.py.
"""

import sys
from pathlib import Path

# numpy is imported for the maths and the log-spaced grids
import numpy as np
# mujoco is imported to change the wrist mass for the payload test
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing what I need from steps 1, 3, 4 and 5
from step4_Sameeha_Work.step1_state_space_model import load_scene, linearise, double_integrator, env, RESULTS  # noqa
from step4_Sameeha_Work.step3_measurement_and_ground_truth import (     # noqa: E402
    run_trial, truth, Rn_for, DELTA_ENC, SIGMA_ENC)
from step4_Sameeha_Work.step4_observers import (                        # noqa: E402
    finite_difference, luenberger_gain, run_luenberger, run_kf, W_O, Q_A)
from step4_Sameeha_Work.step5_evaluation import metrics, nees, K0_S     # noqa: E402

PAYLOAD_KG = 1.0           # need to ask Yara the real object + gripper mass
PUSH_NM, PUSH_T, PUSH_LEN = 10.0, 4.0, 0.3     # 10 N m on the shoulder lift, at t = 4 s, for 0.3 s


def payload_model(extra_kg=PAYLOAD_KG):
    # making a new copy of the scene and adding extra mass to the last wrist link (like holding an object)
    m = env.build_scene()
    m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")] += extra_kg
    return m


def push(t, joint=1, tau=PUSH_NM, t0=PUSH_T, length=PUSH_LEN):
    # This is a 10 N m torque on the shoulder lift only between t0 and t0 + length, 0 the rest of the time
    f = np.zeros(6)
    if t0 <= t < t0 + length:
        f[joint] = tau
    return f


def sustained_load(t, t0=3.0):
    """8 N m on the shoulder lift and 4 N m on the elbow from t0 onwards (stays on)."""
    return np.array([0, 8.0, 4.0, 0, 0, 0]) if t >= t0 else np.zeros(6)


def tuning_set(arm, q_home, T=8.0):
    """The 4 test trials. The observer always uses the NORMAL model at home."""
    # a different start pose to check the model still works away from home
    q_other = q_home + np.array([0.4, -0.3, 0.35, 0.3, -0.4, 0.2])
    # the 4 trials, each with its own noise seed
    return {"nominal": run_trial(arm, q_home, T=T, seed=11),
            "other pose": run_trial(arm, q_other, T=T, seed=12),
            f"payload {PAYLOAD_KG:g} kg": run_trial(arm, q_home, T=T, seed=13, model=payload_model()),
            "sustained load": run_trial(arm, q_home, T=T, seed=14, disturbance=sustained_load)}


def score(X_hat, X, k0, dt, sigma=SIGMA_ENC):
    """Score J (see the top of the file). Returns J, angle RMSE [rad], speed RMSE [rad/s]."""
    from step4_Sameeha_Work.step5_evaluation import metrics
    # This is to calculate the score J: angle error compared to raw encoder noise, plus
    # speed error compared to finite-difference noise. Lower = better, 0 = perfect
    rp, rv, _ = metrics(X_hat, X, k0)
    return rp.mean() / sigma + rv.mean() / (np.sqrt(2) * sigma / dt), rp.mean(), rv.mean()


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    # setting up: scene, model at home, C and the start-up skip
    model, arm, dt, q_home = load_scene()
    Ad, Bd, xs, us = linearise(arm, q_home)
    _, _, C = double_integrator(dt)
    k0 = int(K0_S / dt)
    # recording the 4 tuning trials once, so every setting gets tested on exactly the same data
    trials = tuning_set(arm, q_home)
    truths = {k: truth(v) for k, v in trials.items()}
    print("test situations:", list(trials))

    def sweep(make_estimate, grid):
        """Works out J, angle error, speed error (and NEES if there's a P) for every setting x every trial."""
        # empty tables: one row per setting, one column per trial
        out = {k: np.zeros((len(grid), len(trials))) for k in ["J", "pos", "vel", "nees"]}
        # trying every setting on every trial and saving its score
        for i, g in enumerate(grid):
            for j, (k, tr) in enumerate(trials.items()):
                Xh, Ph = make_estimate(tr, g)
                out["J"][i, j], out["pos"][i, j], out["vel"][i, j] = score(Xh, truths[k], k0, dt)
                out["nees"][i, j] = nees(Xh, Ph, truths[k], k0).mean() if Ph is not None else np.nan
        return out

    # prints a table of the results for each setting
    def table(title, grid, r, gname):
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        print(f"  {gname:>9s}   mean J   worst trial J   angle RMSE [mrad]   speed RMSE [rad/s]   mean NEES")
        for i, g in enumerate(grid):
            print(f"  {g:9.3g}   {r['J'][i].mean():6.3f}   {r['J'][i].max():13.3f}   {1e3 * r['pos'][i].mean():17.3f}"
                  f"   {r['vel'][i].mean():18.4f}   {r['nees'][i].mean():9.2f}")

    # graphs: angle error, speed error (and NEES) vs the setting, with the best one as a dashed line
    def plot(fname, grid, r, xlabel, title, best, ref=None):
        fig, ax = plt.subplots(1, 3 if not np.all(np.isnan(r["nees"])) else 2, figsize=(14, 3.6))
        for j, k in enumerate(trials):
            ax[0].loglog(grid, 1e3 * r["pos"][:, j], "o-", ms=3, lw=1, label=k)
            ax[1].loglog(grid, r["vel"][:, j], "o-", ms=3, lw=1, label=k)
            if len(ax) == 3:
                ax[2].loglog(grid, r["nees"][:, j], "o-", ms=3, lw=1, label=k)
            if ref is not None:
                ax[0].axhline(1e3 * ref["pos"][j], color=f"C{j}", ls=":", lw=1)
                ax[1].axhline(ref["vel"][j], color=f"C{j}", ls=":", lw=1)
        ax[0].axhline(1e3 * SIGMA_ENC, color="k", ls="-.", lw=1, label="raw encoder noise")
        ax[0].set_ylabel("angle RMSE [mrad]"); ax[1].set_ylabel("speed RMSE [rad/s]")
        if len(ax) == 3:
            ax[2].axhline(12, color="k", ls="--", label="n = 12 (honest)"); ax[2].set_ylabel("mean NEES")
        for a in ax:
            a.axvline(best, color="k", ls="--", lw=1); a.set_xlabel(xlabel); a.legend(fontsize=6)
        fig.suptitle(title)
        plt.tight_layout(); fig.savefig(RESULTS / fname, dpi=150)

    # "model only" (L = 0) on each trial - shown as the dotted lines
    ref = {"pos": [], "vel": []}
    for k, tr in trials.items():
        _, p_, v_ = score(run_luenberger(tr, np.zeros((12, 6)), Ad, Bd, xs, us), truths[k], k0, dt)
        ref["pos"].append(p_); ref["vel"].append(v_)

    # ------------------------------------------------------------------ D.1 Luenberger speed
    # nothing to move below ~5 rad/s because the slow servo poles are already at s ~ -5
    # 13 values of w_o from 6 to 400 rad/s, evenly spaced on a log scale
    w_grid = np.logspace(np.log10(6), np.log10(400), 13)
    # MAIN PART: testing the Luenberger observer with every w_o
    lu = sweep(lambda tr, w: (run_luenberger(tr, luenberger_gain(Ad, C, w, dt), Ad, Bd, xs, us), None), w_grid)
    # best w_o = the one with the lowest average J over the 4 trials (this is how I got W_O = 49)
    w_best = w_grid[np.argmin(lu["J"].mean(1))]
    table("D.1 - Luenberger speed w_o", w_grid, lu, "w_o")
    print("  model only: angle RMSE [mrad] per trial", np.round(1e3 * np.array(ref["pos"]), 3),
          "  speed", np.round(ref["vel"], 4))
    print(f"-> best w_o = {w_best:.1f} rad/s")
    plot("step6_luenberger_tuning.png", w_grid, lu, "w_o [rad/s]",
         "D.1 Luenberger: too slow = stuck with the model's error, too fast = lets encoder noise through "
         "(dotted = model only)", w_best, ref)

    # ------------------------------------------------------------------ D.2 Kalman filter q_a
    # 13 values of q_a from 1 to 1,000,000 (log scale)
    qa_grid = np.logspace(0, 6, 13)
    # MAIN PART: testing the KF with every q_a
    kf = sweep(lambda tr, qa: run_kf(tr, qa, Ad, Bd, xs, us), qa_grid)
    # best q_a = lowest average J (this is how I got Q_A = 100)
    qa_best = qa_grid[np.argmin(kf["J"].mean(1))]
    table("D.2 - Kalman filter q_a (Rn is fixed by the encoder, so only Qd/Rn matters)", qa_grid, kf, "q_a")
    i_b = np.argmin(kf["J"].mean(1))
    # checking the best one isn't too confident (NEES not way above 12)
    print(f"-> best q_a = {qa_best:.3g}:  mean NEES {kf['nees'][i_b].mean():.1f}, worst trial "
          f"{kf['nees'][i_b].max():.1f}  ({'honest / a bit cautious: the safe side' if kf['nees'][i_b].max() <= 24 else 'TOO CONFIDENT: raise q_a'})")
    plot("step6_kf_tuning.png", qa_grid, kf, "q_a", "D.2 KF: small q_a trusts the model (wrong under load, "
         "too confident), big q_a trusts the noisy encoder", qa_best, ref)

    # ------------------------------------------------------------------ D.3 encoder quality
    # Each method keeps its tuned setting; the KF's Rn changes with the encoder (Rn belongs to the
    # sensor, it's not something I tune). I look at angle AND speed: with a noisier encoder the KF
    # leans on the model more, which helps speed but keeps the load error in the angle.
    # encoder noise from 0.00001 to 0.01 rad
    sig_grid = np.logspace(-5, -2, 7)
    labels = ["finite difference", "Luenberger", "KF (MuJoCo model)"]
    pos = {k: [] for k in labels}; vel = {k: [] for k in labels}
    # Luenberger gain with the best w_o
    L_best = luenberger_gain(Ad, C, w_best, dt)
    # for each noise level: make a new trial and test the 3 methods on it
    for sg in sig_grid:
        tr = run_trial(arm, q_home, T=6.0, seed=21, sigma=sg, disturbance=sustained_load)
        X = truth(tr)
        for lab, Xh in [("finite difference", finite_difference(tr)),
                        ("Luenberger", run_luenberger(tr, L_best, Ad, Bd, xs, us)),
                        # the KF gets told the new noise level through Rn (Rn belongs to the encoder, it's not tuned)
                        ("KF (MuJoCo model)", run_kf(tr, qa_best, Ad, Bd, xs, us, Rn=Rn_for(sg))[0])]:
            _, p_, v_ = score(Xh, X, k0, dt, sg)
            pos[lab].append(p_); vel[lab].append(v_)
    # graph: errors vs how noisy the encoder is -> step6_noise_sweep.png
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for lab, c in zip(labels, ["0.6", "C2", "C3"]):
        ax[0].loglog(sig_grid, 1e3 * np.array(pos[lab]), "o-", color=c, label=lab)
        ax[1].loglog(sig_grid, vel[lab], "o-", color=c, label=lab)
    ax[0].loglog(sig_grid, 1e3 * sig_grid, "k-.", lw=1, label="raw encoder noise")
    for a in ax:
        a.axvline(SIGMA_ENC, color="k", ls=":", lw=1, label="value used in steps 3-7")
        a.axvline(DELTA_ENC / np.sqrt(12), color="C1", ls=":", lw=1, label="encoder step limit")
        a.set_xlabel("encoder noise sigma [rad]"); a.legend(fontsize=6)
    ax[0].set_ylabel("angle RMSE [mrad]"); ax[1].set_ylabel("speed RMSE [rad/s]")
    fig.suptitle("D.3 accuracy vs how good the encoder is (steady-load trial)")
    plt.tight_layout(); fig.savefig(RESULTS / "step6_noise_sweep.png", dpi=150)

    print("\n" + "=" * 92)
    print("D.3 - encoder quality (steady-load trial): angle [mrad] | speed [rad/s]")
    print("=" * 92)
    print("  sigma [rad]  " + "".join(f"{k:>26s}" for k in labels))
    for i, sg in enumerate(sig_grid):
        print(f"  {sg:9.1e}    " + "".join(f"{1e3 * pos[k][i]:13.3f} | {vel[k][i]:8.4f}" for k in labels))

    # printing the best settings so I can copy them into step4_observers.py and observer.py
    print("\n" + "=" * 78)
    print(f"BEST SETTINGS:  W_O = {w_best:.1f}    Q_A = {qa_best:.3g}")
    print(f"currently in step4_observers.py: W_O = {W_O}, Q_A = {Q_A:g}")
    print("=" * 78)
    print("figures saved to", RESULTS)
    plt.show()