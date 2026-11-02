"""
step8_testing.py -> Task 2 (Sameeha): automatic PASS / FAIL checks for everything I claim

I turned every claim from steps 1-7 and observer.py into a check with a fixed pass mark,
so I can quote the results and re-check them any time I change something.

    Modelling (A)       model matches MuJoCo, holds still at home, stable, controllable, observable
    Sensing (B)         encoder noise matches Rn, results repeat with the same seed
    Estimation (C)      KF beats the raw encoder and finite difference, and is honest (+-2 sigma)
    Robustness (E)      payload, steady load, push, 20 random trials
    Integration         observer.py = step 4, fast enough, works with main.py's update() line

The pass marks come from what the task needs, not from what the code happens to get. The main one
is "angle error below the raw encoder noise" = the observer has to be better than just reading
the encoder.

Run:  python step4_Sameeha_Work/step8_testing.py
      -> PASS/FAIL list in the terminal
      -> results/step8_test_report.md   (same results as a table, for the project page / README)
      -> results/step8_summary.png      (one graph of the errors in every test)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from step4_Sameeha_Work.step1_state_space_model import (                 # noqa: E402
    load_scene, linearise, double_integrator, env, RESULTS)
from step4_Sameeha_Work.step2_controllability_observability import (     # noqa: E402
    rank_ctrb_obsv, pbh_detectable_d, pbh_stabilisable_d)
from step4_Sameeha_Work.step3_measurement_and_ground_truth import (      # noqa: E402
    run_trial, truth, encoder, reference, Rn_for, SIGMA_ENC)
from step4_Sameeha_Work.step4_observers import (                         # noqa: E402
    finite_difference, luenberger_gain, run_luenberger, run_kf, rmse, W_O, Q_A)
from step4_Sameeha_Work.step5_evaluation import metrics, nees, K0_S     # noqa: E402
from step4_Sameeha_Work.step6_tuning import payload_model, push, sustained_load, PUSH_T, PUSH_LEN  # noqa
from step4_Sameeha_Work.observer import JointObserver, add_encoder_sensors   # noqa: E402

warnings.filterwarnings("ignore")                     # hides ct.place warnings
CHECKS = []                                           # list of (area, check, result, pass mark, passed?)


def check(area, name, measured, mark, ok):
    CHECKS.append((area, name, measured, mark, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} {measured:>22s}   (pass: {mark})")


def section(title):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)


def estimates(tr, lin, L):
    """The methods I compare: finite difference, model only (no encoder), Luenberger, Kalman filter."""
    Ad, Bd, xs, us = lin
    X_kf, P_kf = run_kf(tr, Q_A, Ad, Bd, xs, us)
    return {"finite difference": finite_difference(tr),
            "model only": run_luenberger(tr, np.zeros_like(L), Ad, Bd, xs, us),
            "Luenberger": run_luenberger(tr, L, Ad, Bd, xs, us),
            "Kalman filter": X_kf}, P_kf


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    RESULTS.mkdir(exist_ok=True)
    t_start = time.time()

    model, arm, dt, q_home = load_scene()
    lin = linearise(arm, q_home)
    Ad, Bd, x_star, u_star = lin
    Ad_di, Bd_di, C = double_integrator(dt)
    L = luenberger_gain(Ad, C, W_O, dt)
    k0 = int(K0_S / dt)
    enc_mrad = 1e3 * np.sqrt(Rn_for()[0, 0])          # raw encoder noise = what I have to beat
    print(f"Task 2 tests | encoder noise {enc_mrad:.3f} mrad | W_O = {W_O}, Q_A = {Q_A:g} | dt = {dt} s")

    # ------------------------------------------------------------------ A. modelling
    section("A. MODELLING (criterion A)")
    du = 0.05 * np.ones(6)
    d = mujoco.MjData(model)
    d.qpos[arm.qadr] = q_home
    mujoco.mj_forward(model, d)
    X_nl, X_lin, dx = [x_star.copy()], [x_star.copy()], np.zeros(12)
    for _ in range(int(1.0 / dt)):
        d.ctrl[arm.act] = u_star + du
        mujoco.mj_step(model, d)
        dx = Ad @ dx + Bd @ du
        X_nl.append(np.r_[d.qpos[arm.qadr], d.qvel[arm.dof]]); X_lin.append(x_star + dx)
    X_nl, X_lin = np.array(X_nl), np.array(X_lin)
    fit = 100 * (1 - np.linalg.norm(X_nl - X_lin) / np.linalg.norm(X_nl - X_nl.mean(0)))
    check("A", "my linear model matches MuJoCo (1 s step test)", f"{fit:.2f} %", ">= 95 %", fit >= 95)

    d = mujoco.MjData(model)
    env.reset_home(model, d)
    d.ctrl[arm.act] = u_star
    for _ in range(int(1.0 / dt)):
        mujoco.mj_step(model, d)
    drift = np.abs(d.qpos[arm.qadr] - q_home).max()
    check("A", "arm holds still at home with u* (drift after 1 s)", f"{drift:.1e} rad", "< 1e-6 rad", drift < 1e-6)

    zmax = np.abs(np.linalg.eigvals(Ad)).max()
    check("A", "model is stable (biggest |pole|)", f"{zmax:.4f}", "< 1", zmax < 1)

    for name, A_, B_ in [("double integrator", Ad_di, Bd_di), ("MuJoCo model", Ad, Bd)]:
        rc, ro = rank_ctrb_obsv(A_, B_, C)
        check("A", f"controllable + observable with encoders ({name})", f"ranks {rc}, {ro}", "both 12",
              rc == 12 and ro == 12)
    ok = pbh_detectable_d(Ad, C)[0] and pbh_stabilisable_d(Ad, Bd)[0]
    check("A", "detectable + stabilisable (PBH) -> separation OK", str(ok), "True", ok)

    # ------------------------------------------------------------------ B. sensing
    section("B. SENSING (criterion B)")
    rng = np.random.default_rng(0)
    s = np.array([encoder(q_home, rng) - q_home for _ in range(20000)]).std()
    rel = abs(s - np.sqrt(Rn_for()[0, 0])) / np.sqrt(Rn_for()[0, 0])
    check("B", "encoder noise matches the Rn the filter uses", f"{100 * rel:.1f} % off", "within 5 %", rel < 0.05)
    a, b = run_trial(arm, q_home, T=1.0, seed=3), run_trial(arm, q_home, T=1.0, seed=3)
    same = np.array_equal(a["y"], b["y"]) and np.array_equal(a["q"], b["q"])
    check("B", "same seed gives exactly the same data", str(same), "True", same)

    # ------------------------------------------------------------------ C. estimation (nominal trial)
    section("C. ESTIMATION, normal trial (criterion C)")
    tr = run_trial(arm, q_home, T=10.0, seed=1)
    X = truth(tr)
    est, P_kf = estimates(tr, lin, L)
    res = {m: metrics(Xh, X, k0) for m, Xh in est.items()}
    kf_a, kf_v = 1e3 * res["Kalman filter"][0].mean(), res["Kalman filter"][1].mean()
    fd_v, lu_v = res["finite difference"][1].mean(), res["Luenberger"][1].mean()
    check("C", "KF angle error is below the raw encoder noise", f"{kf_a:.3f} mrad", f"< {enc_mrad:.3f} mrad", kf_a < enc_mrad)
    check("C", "KF speed error at least 50x better than finite diff.", f"{fd_v / kf_v:.0f}x", ">= 50x",
          fd_v / kf_v >= 50)
    check("C", "KF speed error lower than Luenberger", f"{kf_v:.4f} vs {lu_v:.4f}", "KF lower", kf_v < lu_v)
    worst_joint = res["Kalman filter"][1].max()
    check("C", "every joint's speed error is small (worst joint)", f"{worst_joint:.4f} rad/s", "< 0.01 rad/s",
          worst_joint < 0.01)
    sig = np.sqrt(np.einsum("kii->ki", P_kf))
    inside = 100 * np.mean(np.abs(est["Kalman filter"][k0:] - X[k0:]) < 2 * sig[k0:])
    check("C", "KF errors stay inside its own +-2 sigma band", f"{inside:.1f} %", ">= 95 %", inside >= 95)

    # ------------------------------------------------------------------ E. robustness
    section("E. ROBUSTNESS (criterion E) - the observers always use the NORMAL model")
    scen = {"nominal": tr,
            "other pose": run_trial(arm, q_home + np.array([0.4, -0.3, 0.35, 0.3, -0.4, 0.2]), T=8, seed=12),
            "0.5 kg payload": run_trial(arm, q_home, T=8, seed=41, model=payload_model(0.5)),
            "1.0 kg payload": run_trial(arm, q_home, T=8, seed=41, model=payload_model(1.0)),
            "1.5 kg payload": run_trial(arm, q_home, T=8, seed=41, model=payload_model(1.5)),
            "steady load": run_trial(arm, q_home, T=8, seed=14, disturbance=sustained_load),
            "push": run_trial(arm, q_home, T=8, seed=31, disturbance=push)}
    table = {}                                                       # test -> method -> (angle, speed)
    for name, s_tr in scen.items():
        e, _ = estimates(s_tr, lin, L)
        table[name] = {m: (1e3 * metrics(Xh, truth(s_tr), k0)[0].mean(), metrics(Xh, truth(s_tr), k0)[1].mean())
                       for m, Xh in e.items()}
    for name in ["other pose", "1.5 kg payload", "steady load", "push"]:
        a_kf = table[name]["Kalman filter"][0]
        check("E", f"KF angle error below encoder noise: {name}", f"{a_kf:.3f} mrad", f"< {enc_mrad:.3f} mrad",
              a_kf < enc_mrad)
    better = all(table[n]["Kalman filter"][0] < table[n]["model only"][0]
                 for n in ["0.5 kg payload", "1.0 kg payload", "1.5 kg payload", "steady load"])
    check("E", "encoders help: KF beats 'model only' with any load", str(better), "True", better)

    pt = scen["push"]
    e_push, _ = estimates(pt, lin, L)
    after = slice(int((PUSH_T + PUSH_LEN + 0.5) / dt), int((PUSH_T + PUSH_LEN + 1.5) / dt))
    rec = 1e3 * rmse(e_push["Kalman filter"][after, :6] - truth(pt)[after, :6]).mean()
    check("E", "KF recovers after the push (0.5-1.5 s later)", f"{rec:.3f} mrad", f"< {enc_mrad:.3f} mrad",
          rec < enc_mrad)

    rng = np.random.default_rng(7)
    mc = []
    for i in range(20):
        m_tr = run_trial(arm, q_home + rng.uniform(-0.5, 0.5, 6), T=6, seed=100 + i,
                         model=payload_model(rng.uniform(0, 1)))
        Xh, _ = run_kf(m_tr, Q_A, Ad, Bd, x_star, u_star)
        mc.append(1e3 * metrics(Xh, truth(m_tr), k0)[0].mean())
    check("E", "20 random trials (pose + payload): worst KF angle", f"{max(mc):.3f} mrad",
          f"< {enc_mrad:.3f} mrad", max(mc) < enc_mrad)

    # ------------------------------------------------------------------ integration: observer.py
    section("INTEGRATION - observer.py (the file the team will use)")
    o = JointObserver(model, add_noise=False)
    t_short = run_trial(arm, q_home, T=3.0, seed=5)
    X_obs = np.array([np.r_[t_short["y"][0], np.zeros(6)]] +
                     [o.step(t_short["y"][k - 1], t_short["u"][k - 1])[0] for k in range(1, len(t_short["t"]))])
    X_ref, _ = run_kf(t_short, Q_A, Ad, Bd, x_star, u_star)
    diff = np.abs(X_obs - X_ref).max()
    check("I", "observer.py gives the same answers as step 4", f"{diff:.1e}", "< 1e-9", diff < 1e-9)

    # same loop as main.py: scene + encoder sensors (added here only, environment.py isn't changed)
    _compile = mujoco.MjSpec.compile
    mujoco.MjSpec.compile = lambda spec: _compile(add_encoder_sensors(spec))
    try:
        m_s = env.build_scene()
    finally:
        mujoco.MjSpec.compile = _compile
    d = mujoco.MjData(m_s)
    env.reset_home(m_s, d)
    obs = JointObserver(m_s, seed=0)
    err, costs = [], []
    for _ in range(int(4.0 / dt)):
        t0 = time.perf_counter()
        x_hat, P = obs.update(d.sensordata, d.ctrl)                  # exactly the line in main.py
        costs.append(time.perf_counter() - t0)
        err.append(x_hat - np.r_[d.qpos[arm.qadr], d.qvel[arm.dof]])
        qr, qdr, qddr = reference(d.time, q_home)
        d.ctrl[arm.act] = arm.ctrl_for_force(arm.inverse_dynamics(qr, qdr, qddr), qr, qdr)
        mujoco.mj_step(m_s, d)
    err = np.array(err)[k0:]
    a_loop = 1e3 * rmse(err[:, :6]).mean()
    check("I", "works with main.py's update(sensordata, ctrl) line", f"{a_loop:.3f} mrad",
          f"< {enc_mrad:.3f} mrad", a_loop < enc_mrad)
    us = 1e6 * np.median(costs)
    check("I", "fast enough for real time (one update vs 2 ms)", f"{us:.0f} us", "< 2000 us", us < 2000)

    try:
        JointObserver(model).update(np.zeros(0), np.zeros(6))
        ok = False
    except RuntimeError:
        ok = True
    check("I", "clear error message if there are no encoder sensors", str(ok), "True", ok)

    # ------------------------------------------------------------------ save the report + summary graph
    n_pass = sum(c[4] for c in CHECKS)
    section(f"RESULT: {n_pass} / {len(CHECKS)} checks passed   ({time.time() - t_start:.0f} s)")

    areas = {"A": "Modelling (A)", "B": "Sensing (B)", "C": "Estimation (C)", "E": "Robustness (E)",
             "I": "Integration"}
    lines = ["# Task 2 test report", "",
             f"Generated by `step8_testing.py`: **{n_pass} / {len(CHECKS)} checks passed**. "
             f"Encoder noise {enc_mrad:.3f} mrad, W_O = {W_O}, Q_A = {Q_A:g}, dt = {dt} s.", "",
             "| Area | Check | Measured | Pass mark | Result |", "|---|---|---|---|---|"]
    lines += [f"| {areas[a]} | {n} | {m} | {mk} | {'PASS' if ok else '**FAIL**'} |" for a, n, m, mk, ok in CHECKS]
    lines += ["", "## Angle error per scenario [mrad] (raw encoder noise = %.3f)" % enc_mrad, "",
              "| Scenario | model only | Luenberger | Kalman filter |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['model only'][0]:.3f} | {v['Luenberger'][0]:.3f} | {v['Kalman filter'][0]:.3f} |"
              for s, v in table.items()]
    lines += ["", "## Velocity error per scenario [rad/s]", "",
              "| Scenario | finite difference | Luenberger | Kalman filter |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['finite difference'][1]:.4f} | {v['Luenberger'][1]:.4f} | {v['Kalman filter'][1]:.4f} |"
              for s, v in table.items()]
    lines += ["", f"20 randomised trials (pose +-0.5 rad, payload 0-1 kg): KF angle error median "
              f"{np.median(mc):.3f} mrad, worst {max(mc):.3f} mrad.", ""]
    (RESULTS / "step8_test_report.md").write_text("\n".join(lines), encoding="utf-8")

    names = list(table)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    for i, (m, c) in enumerate([("model only", "C1"), ("Luenberger", "C2"), ("Kalman filter", "C3")]):
        ax[0].bar(x + 0.27 * (i - 1), [table[n][m][0] for n in names], 0.27, color=c, label=m)
    ax[0].axhline(enc_mrad, color="k", ls="-.", lw=1.2, label="raw encoder noise")
    ax[0].set_ylim(top=1.25 * max(table[n]["model only"][0] for n in names))
    ax[0].set(ylabel="angle error [mrad]", title="Joint angles: my estimate vs MuJoCo truth (lower = better)")
    for i, (m, c) in enumerate([("finite difference", "0.6"), ("Luenberger", "C2"), ("Kalman filter", "C3")]):
        ax[1].bar(x + 0.27 * (i - 1), [table[n][m][1] for n in names], 0.27, color=c, label=m)
    ax[1].set_yscale("log"); ax[1].set_ylim(top=5)                 # space at the top for the legend
    ax[1].set(ylabel="speed error [rad/s] (log scale)", title="Joint speeds (not measured): my estimate vs truth")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=20, fontsize=8)
        a.legend(fontsize=8, loc="upper center", ncol=4)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Task 2 summary: {n_pass}/{len(CHECKS)} checks passed", fontweight="bold")
    plt.tight_layout(); fig.savefig(RESULTS / "step8_summary.png", dpi=150)

    print("report -> results/step8_test_report.md   figure -> results/step8_summary.png")
    plt.show()
    sys.exit(0 if n_pass == len(CHECKS) else 1)
