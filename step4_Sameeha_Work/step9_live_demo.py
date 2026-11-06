"""
step9_live_demo.py -> Task 2 (Sameeha): watch my observer working live in the MuJoCo viewer

Opens the MuJoCo window, moves the arm, and runs my Kalman filter (observer.py) every step,
like it will in main.py. The terminal prints how close my estimate is to the true values.

What happens (repeats every 15 s):
    0 - 5 s    normal motion
    5 s        a 10 N m push on the shoulder lift (the observer isn't told)
    8 - 15 s   an extra 1 kg on the wrist, like a picked-up object (the observer isn't told)

Close the window to stop. Then a graph of the whole run is saved to results/step9_live_demo.png.

Nothing in environment.py or main.py is changed: the encoder sensors are added only inside this file.
This one isn't in run_all.py because it needs the window.

Run:  press the play button, or  python step4_Sameeha_Work/step9_live_demo.py
"""

import sys
# time is imported to make the simulation run in real time
import time
from pathlib import Path

import numpy as np
import mujoco
# mujoco.viewer is imported to open the 3D window
import mujoco.viewer

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing the arm, the test motion, the encoder noise and my observer
from step4_Sameeha_Work.step1_state_space_model import Arm, env, NAMES, RESULTS        # noqa: E402
from step4_Sameeha_Work.step3_measurement_and_ground_truth import reference, Rn_for     # noqa: E402
from step4_Sameeha_Work.observer import JointObserver, add_encoder_sensors              # noqa: E402

LOOP = 15.0                     # s, the scenario repeats every 15 s
# the push: starts at 5 s, lasts 0.3 s, 10 N m
PUSH_AT, PUSH_LEN, PUSH_NM = 5.0, 0.3, 10.0
# the payload: 1 kg added from 8 s
PAYLOAD_FROM, PAYLOAD_KG = 8.0, 1.0


def phase(t):
    # This is to work out what's happening at time t (the scenario repeats every 15 s)
    s = t % LOOP
    if PUSH_AT <= s < PUSH_AT + PUSH_LEN:
        return "PUSH"
    return "1 kg payload" if s >= PAYLOAD_FROM else "normal"


def scene_with_encoders():
    """Same scene as environment.build_scene(), plus the 6 encoder sensors (only for this demo)."""
    # temporarily changing compile so build_scene() adds my 6 encoder sensors, then putting it back
    _compile = mujoco.MjSpec.compile
    mujoco.MjSpec.compile = lambda spec: _compile(add_encoder_sensors(spec))
    try:
        return env.build_scene()
    finally:
        mujoco.MjSpec.compile = _compile


if __name__ == "__main__":
    # building the scene with encoders, putting the arm at home and the sphere somewhere random
    model = scene_with_encoders()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    env.randomise_object(model, data, np.random.default_rng())

    observer = JointObserver(model, seed=0)               # my Kalman filter, same as main.py would use
    arm = observer.arm
    planner = Arm(env.build_scene())                      # normal robot model for the stand-in controller,
    q_home = observer.x_star[:6].copy()                   # so it doesn't know about the payload either
    # finding the wrist link so I can add the payload mass to it
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    m0 = model.body_mass[wrist]
    # raw encoder noise in mrad (about 1.0) = what my estimate has to beat
    enc_mrad = 1e3 * np.sqrt(Rn_for()[0, 0])

    # lists to save the time, truth, estimate and KF uncertainty every step
    log_t, log_true, log_est, log_sig, seen = [], [], [], [], set()
    next_print = 1.0
    print(f"raw encoder noise = {enc_mrad:.2f} mrad.  My estimate should stay below that.  Close the window to stop.\n")
    print("   time   what's happening    angle error [mrad]   speed error [rad/s]   KF says +-2 sigma [rad/s]")
    print("                                (actual error vs truth, average over the last second)   (its own uncertainty)")

    # opening the MuJoCo window and running until it's closed
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time
            ph = phase(t)
            seen.add(ph)                                  # what happened during this second

            # ---- what the observer doesn't know about
            # adding the payload mass during the payload part
            model.body_mass[wrist] = m0 + (PAYLOAD_KG if ph == "1 kg payload" else 0.0)
            # adding the push torque during the push part
            data.qfrc_applied[arm.dof] = [0, PUSH_NM, 0, 0, 0, 0] if ph == "PUSH" else 0

            # ---- my part: exactly the line from main.py
            # MAIN PART: my Kalman filter gives the estimated 12 states (x_hat) and its uncertainty (P)
            x_hat, P = observer.update(data.sensordata, data.ctrl)

            # saving the time, the TRUE values from MuJoCo (only for checking) and my estimate
            log_t.append(t)
            log_true.append(np.r_[data.qpos[arm.qadr], data.qvel[arm.dof]])
            log_est.append(x_hat)
            log_sig.append(np.sqrt(np.diag(P)))           # the KF's own uncertainty (1 sigma) for each state

            # ---- stand-in controller (Yara's goes here): follows the test motion
            qr, qdr, qddr = reference(t, q_home)
            data.ctrl[arm.act] = planner.ctrl_for_force(planner.inverse_dynamics(qr, qdr, qddr), qr, qdr)

            # one physics step, then updating the window
            mujoco.mj_step(model, data)
            viewer.sync()

            # ---- print every second
            if data.time >= next_print:
                next_print += 1.0
                n = int(1.0 / model.opt.timestep)
                # the error of the last second of estimates vs the truth
                E = np.array(log_est[-n:]) - np.array(log_true[-n:])
                # average angle error (mrad) and speed error (rad/s) over that second
                a_err = 1e3 * np.sqrt((E[:, :6] ** 2).mean())
                v_err = np.sqrt((E[:, 6:] ** 2).mean())
                label = "PUSH" if "PUSH" in seen else ("1 kg payload" if "1 kg payload" in seen else "normal")
                seen = set()
                # the KF's own +-2 sigma for the speeds, averaged over that second
                s2 = 2 * np.array(log_sig[-n:])[:, 6:].mean()
                print(f"  {data.time:5.1f} s   {label:16s}   {a_err:10.3f}  {'OK' if a_err < enc_mrad else '  '}"
                      f"          {v_err:10.4f}             {s2:10.4f}")

            dt = model.opt.timestep - (time.time() - step_start)       # run in real time
            if dt > 0:
                time.sleep(dt)

    # ---- graph of the whole run
    if len(log_t) > 10:
        import matplotlib.pyplot as plt
        RESULTS.mkdir(exist_ok=True)
        # turning the lists into arrays for plotting
        tt, X, Xh, S = np.array(log_t), np.array(log_true), np.array(log_est), np.array(log_sig)
        # 3 graphs on top of each other sharing the time axis
        fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        # 1) the estimate alongside the true state
        ax[0].plot(tt, X[:, 7], "k", lw=2, label="true speed (MuJoCo)")
        ax[0].plot(tt, Xh[:, 7], "C3--", lw=1.2, label="my estimate (never measured)")
        ax[0].set_ylabel("shoulder lift speed [rad/s]"); ax[0].legend(fontsize=8, loc="upper right")
        # 2) speed error inside the KF's own +-2 sigma band (the band is too thin to see on the plot above)
        ax[1].fill_between(tt, -2 * S[:, 7], 2 * S[:, 7], color="C0", alpha=0.3, label="KF's own +-2 sigma (uncertainty)")
        ax[1].plot(tt, Xh[:, 7] - X[:, 7], "C3", lw=0.6, label="actual error (estimate - truth)")
        k_s = min(len(tt) - 1, int(0.5 / model.opt.timestep))          # skip the first 0.5 s (filter starting up)
        ax[1].set_ylim(-4 * np.median(S[k_s:, 7]), 4 * np.median(S[k_s:, 7]))
        ax[1].set_ylabel("lift speed error [rad/s]"); ax[1].legend(fontsize=8, loc="upper right")
        # 3) angle error inside its +-2 sigma band, compared with the raw encoder noise
        ax[2].fill_between(tt, -2e3 * S[:, 1], 2e3 * S[:, 1], color="C0", alpha=0.3, label="KF's own +-2 sigma")
        ax[2].plot(tt, 1e3 * (Xh[:, 1] - X[:, 1]), "C3", lw=0.6, label="actual error")
        ax[2].axhline(enc_mrad, color="k", ls="-.", lw=1, label="raw encoder noise")
        ax[2].axhline(-enc_mrad, color="k", ls="-.", lw=1)
        ax[2].set_ylim(-2.5 * enc_mrad, 2.5 * enc_mrad)
        ax[2].set_ylabel("lift angle error [mrad]"); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8, loc="upper right")
        # % of all errors inside +-2 sigma (skipping the first 1 s = 500 steps)
        inside = 100 * np.mean(np.abs(Xh[500:] - X[500:]) < 2 * S[500:])
        print(f"errors inside the KF's own +-2 sigma: {inside:.1f} %")
        # shading: orange for the push, light blue for the payload part
        for a in ax:
            for k in range(int(tt[-1] // LOOP) + 1):
                if k * LOOP + PUSH_AT < tt[-1]:
                    a.axvspan(k * LOOP + PUSH_AT, min(k * LOOP + PUSH_AT + PUSH_LEN, tt[-1]), color="C1", alpha=0.3)
                if k * LOOP + PAYLOAD_FROM < tt[-1]:
                    a.axvspan(k * LOOP + PAYLOAD_FROM, min((k + 1) * LOOP, tt[-1]), color="C0", alpha=0.08)
            a.set_xlim(0, tt[-1]); a.grid(alpha=0.3)
        fig.suptitle(f"Live demo: estimate + uncertainty vs truth. Orange = push, blue = 1 kg payload "
                     f"(observer not told). {inside:.0f} % of errors inside +-2 sigma")
        plt.tight_layout(); fig.savefig(RESULTS / "step9_live_demo.png", dpi=150)
        print("\ngraph saved to results/step9_live_demo.png")
        plt.show()