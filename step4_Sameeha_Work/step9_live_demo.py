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
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from step4_Sameeha_Work.step1_state_space_model import Arm, env, NAMES, RESULTS        # noqa: E402
from step4_Sameeha_Work.step3_measurement_and_ground_truth import reference, Rn_for     # noqa: E402
from step4_Sameeha_Work.observer import JointObserver, add_encoder_sensors              # noqa: E402

LOOP = 15.0                     # s, the scenario repeats every 15 s
PUSH_AT, PUSH_LEN, PUSH_NM = 5.0, 0.3, 10.0
PAYLOAD_FROM, PAYLOAD_KG = 8.0, 1.0


def phase(t):
    s = t % LOOP
    if PUSH_AT <= s < PUSH_AT + PUSH_LEN:
        return "PUSH"
    return "1 kg payload" if s >= PAYLOAD_FROM else "normal"


def scene_with_encoders():
    """Same scene as environment.build_scene(), plus the 6 encoder sensors (only for this demo)."""
    _compile = mujoco.MjSpec.compile
    mujoco.MjSpec.compile = lambda spec: _compile(add_encoder_sensors(spec))
    try:
        return env.build_scene()
    finally:
        mujoco.MjSpec.compile = _compile


if __name__ == "__main__":
    model = scene_with_encoders()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    env.randomise_object(model, data, np.random.default_rng())

    observer = JointObserver(model, seed=0)               # my Kalman filter, same as main.py would use
    arm = observer.arm
    planner = Arm(env.build_scene())                      # normal robot model for the stand-in controller,
    q_home = observer.x_star[:6].copy()                   # so it doesn't know about the payload either
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    m0 = model.body_mass[wrist]
    enc_mrad = 1e3 * np.sqrt(Rn_for()[0, 0])

    log_t, log_true, log_est, seen = [], [], [], set()
    next_print = 1.0
    print(f"raw encoder noise = {enc_mrad:.2f} mrad.  My estimate should stay below that.  Close the window to stop.\n")
    print("   time   what's happening    angle error [mrad]   speed error [rad/s]   (average over the last second)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time
            ph = phase(t)
            seen.add(ph)                                  # what happened during this second

            # ---- what the observer doesn't know about
            model.body_mass[wrist] = m0 + (PAYLOAD_KG if ph == "1 kg payload" else 0.0)
            data.qfrc_applied[arm.dof] = [0, PUSH_NM, 0, 0, 0, 0] if ph == "PUSH" else 0

            # ---- my part: exactly the line from main.py
            x_hat, P = observer.update(data.sensordata, data.ctrl)

            log_t.append(t)
            log_true.append(np.r_[data.qpos[arm.qadr], data.qvel[arm.dof]])
            log_est.append(x_hat)

            # ---- stand-in controller (Yara's goes here): follows the test motion
            qr, qdr, qddr = reference(t, q_home)
            data.ctrl[arm.act] = planner.ctrl_for_force(planner.inverse_dynamics(qr, qdr, qddr), qr, qdr)

            mujoco.mj_step(model, data)
            viewer.sync()

            # ---- print every second
            if data.time >= next_print:
                next_print += 1.0
                n = int(1.0 / model.opt.timestep)
                E = np.array(log_est[-n:]) - np.array(log_true[-n:])
                a_err = 1e3 * np.sqrt((E[:, :6] ** 2).mean())
                v_err = np.sqrt((E[:, 6:] ** 2).mean())
                label = "PUSH" if "PUSH" in seen else ("1 kg payload" if "1 kg payload" in seen else "normal")
                seen = set()
                print(f"  {data.time:5.1f} s   {label:16s}   {a_err:10.3f}  {'OK' if a_err < enc_mrad else '  '}"
                      f"          {v_err:10.4f}")

            dt = model.opt.timestep - (time.time() - step_start)       # run in real time
            if dt > 0:
                time.sleep(dt)

    # ---- graph of the whole run
    if len(log_t) > 10:
        import matplotlib.pyplot as plt
        RESULTS.mkdir(exist_ok=True)
        tt, X, Xh = np.array(log_t), np.array(log_true), np.array(log_est)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(tt, X[:, 7], "k", lw=2, label="true speed (MuJoCo)")
        ax[0].plot(tt, Xh[:, 7], "C3--", lw=1.2, label="my estimate (never measured)")
        ax[0].set_ylabel("shoulder lift speed [rad/s]"); ax[0].legend(fontsize=8)
        ax[1].plot(tt, 1e3 * np.abs(Xh[:, :6] - X[:, :6]).mean(1), "C3", lw=0.6, label="my angle error")
        ax[1].axhline(enc_mrad, color="k", ls="-.", label="raw encoder noise")
        ax[1].set_ylabel("average angle error [mrad]"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8)
        for a in ax:
            for k in range(int(tt[-1] // LOOP) + 1):
                if k * LOOP + PUSH_AT < tt[-1]:
                    a.axvspan(k * LOOP + PUSH_AT, min(k * LOOP + PUSH_AT + PUSH_LEN, tt[-1]), color="C1", alpha=0.3)
                if k * LOOP + PAYLOAD_FROM < tt[-1]:
                    a.axvspan(k * LOOP + PAYLOAD_FROM, min((k + 1) * LOOP, tt[-1]), color="C0", alpha=0.08)
            a.set_xlim(0, tt[-1]); a.grid(alpha=0.3)
        fig.suptitle("Live demo: orange = push, blue = 1 kg payload (the observer isn't told about either)")
        plt.tight_layout(); fig.savefig(RESULTS / "step9_live_demo.png", dpi=150)
        print("\ngraph saved to results/step9_live_demo.png")
        plt.show()
