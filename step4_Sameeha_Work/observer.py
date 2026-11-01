"""
observer.py -> Task 2 (Sameeha): my finished Kalman filter, ready for main.py to import

What's in here:
    JointObserver           Kalman filter: 6 noisy encoders in -> 12 states out (6 angles + 6 speeds)
    add_encoder_sensors()   adds the 6 encoder sensors to the scene (only used by the test at the bottom).
                            environment.py can't import this file (it would be a circular import),
                            so it adds the sensors itself - the snippet is in README.md

Built from step 1 (model), step 3 (encoder noise) and step 4 (Kalman filter), tuned in step 6.
Only needs numpy + mujoco, so main.py doesn't need any new packages.

How to use it in main.py (every physics step, BEFORE the controller sets data.ctrl):

    from step4_Sameeha_Work.observer import JointObserver
    observer = JointObserver(model)
    ...
    x_hat, P = observer.update(data.sensordata, data.ctrl)
    q_hat, qd_hat = x_hat[:6], x_hat[6:]

Run this file to test it in a copy of the main.py loop (no viewer):
    python step4_Sameeha_Work/observer.py
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from step4_Sameeha_Work.step1_state_space_model import (                 # noqa: E402
    Arm, UR_JOINTS, linearise, make_Qd)
from step4_Sameeha_Work.step3_measurement_and_ground_truth import (      # noqa: E402
    SIGMA_ENC, DELTA_ENC, encoder, Rn_for)

ENC_PREFIX = "enc_"
Q_A = 100.0          # Kalman filter setting from step6_tuning.py (if it changes there, change it here too)


def add_encoder_sensors(spec):
    """Adds one angle sensor per arm joint to the scene (has to be before spec.compile())."""
    for j in UR_JOINTS:
        spec.add_sensor(name=ENC_PREFIX + j, type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=j)
    return spec


class JointObserver:
    """Kalman filter using my MuJoCo model of the UR3e (same as step 4).

    model        the scene main.py uses
    q0           pose to linearise around (default = home)
    add_noise    True = add encoder noise to MuJoCo's perfect reading first (step 3), so nothing
                 downstream ever gets perfect data
    relin_every  re-linearise every N updates (tested in step 7, E.4); None = keep one model
    """

    def __init__(self, model, q0=None, q_a=Q_A, sigma_enc=SIGMA_ENC, delta_enc=DELTA_ENC,
                 add_noise=True, seed=None, relin_every=None):
        self.arm = arm = Arm(model)
        self.n = n = arm.n
        self.dt = model.opt.timestep
        if q0 is None:
            q0 = model.key_qpos[0][arm.qadr] if model.nkey > 0 else np.zeros(n)
        self.C = np.hstack([np.eye(n), np.zeros((n, n))])
        self.Qd = make_Qd(q_a, self.dt, n)
        self.Rn = Rn_for(sigma_enc, delta_enc, n)
        self.sigma_enc, self.delta_enc = sigma_enc, delta_enc
        self.add_noise, self.rng = add_noise, np.random.default_rng(seed)
        self.relin_every = relin_every
        self.Ad, self.Bd, self.x_star, self.u_star = linearise(arm, np.asarray(q0, float))
        sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, ENC_PREFIX + j) for j in UR_JOINTS]
        self.sensor_adr = model.sensor_adr[sid] if min(sid) >= 0 else None
        self.reset()

    def reset(self):
        self.xh = None
        self.P = np.diag(np.r_[np.full(self.n, self.Rn[0, 0]), np.full(self.n, 0.5**2)])
        self.k = 0

    # ---------------------------------------------------------------- what main.py calls
    def update(self, sensordata, ctrl):
        """Takes data.sensordata and data.ctrl, gives back (x_hat, P) = estimate of the state right now.

        After mj_step, sensordata is the angle from the START of the last step and ctrl is the
        command used during it (see step 3). So I correct with that reading, then predict one step on."""
        if self.sensor_adr is None:
            raise RuntimeError("The scene has no encoder sensors yet. Add the 6 'enc_<joint name>' "
                               "sensors in environment.build_scene() before spec.compile() "
                               "(the snippet is in step4_Sameeha_Work/README.md).")
        q = np.asarray(sensordata)[self.sensor_adr]
        y = encoder(q, self.rng, self.sigma_enc, self.delta_enc) if self.add_noise else q
        return self.step(y, np.asarray(ctrl)[self.arm.act])

    def step(self, y_prev, u_prev):
        """One update: last encoder reading + the command used since then."""
        y_prev, u_prev = np.asarray(y_prev, float), np.asarray(u_prev, float)
        if self.xh is None:                       # first time: angles from the encoders, speeds = 0
            self.xh = np.r_[y_prev, np.zeros(self.n)]
        else:
            self._correct(y_prev)
        self._predict(u_prev)
        return self.xh.copy(), self.P.copy()

    # ---------------------------------------------------------------- Kalman filter steps (from step 4)
    def _correct(self, y):
        C = self.C
        S = C @ self.P @ C.T + self.Rn
        K = np.linalg.solve(S, C @ self.P).T      # = P C' S^-1  (S symmetric)
        self.xh = self.xh + K @ (y - C @ self.xh)
        self.P = (np.eye(2 * self.n) - K @ C) @ self.P

    def _predict(self, u):
        self.k += 1
        if self.relin_every and self.k % self.relin_every == 0:
            self.Ad, self.Bd, self.x_star, self.u_star = linearise(self.arm, self.xh[:self.n])
        A = self.Ad
        self.xh = self.x_star + A @ (self.xh - self.x_star) + self.Bd @ (u - self.u_star)
        self.P = A @ self.P @ A.T + self.Qd

    @property
    def q_hat(self):
        return self.xh[:self.n]

    @property
    def qd_hat(self):
        return self.xh[self.n:]

    @property
    def sigma(self):
        """How unsure it is about each of the 12 states (1 sigma)."""
        return np.sqrt(np.diag(self.P))


# ================================================================================ run
if __name__ == "__main__":
    # Copy of the main.py loop (no viewer) with my observer lines switched on.
    # I add the encoder sensors here just for this test - environment.py isn't changed.
    import step1_Main.environment as env
    from step4_Sameeha_Work.step3_measurement_and_ground_truth import reference

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    _compile = mujoco.MjSpec.compile

    def _compile_with_encoders(spec):           # same scene as build_scene(), plus the 6 encoders
        add_encoder_sensors(spec)
        return _compile(spec)

    mujoco.MjSpec.compile = _compile_with_encoders
    try:
        model = env.build_scene()
    finally:
        mujoco.MjSpec.compile = _compile
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    env.randomise_object(model, data, np.random.default_rng(0))

    observer = JointObserver(model, seed=0)
    arm = observer.arm
    q_home = observer.x_star[:6].copy()
    print("scene has", model.nsensor, "sensors; encoders at sensordata", observer.sensor_adr)

    T = 6.0
    err = []
    for k in range(int(T / model.opt.timestep)):
        # ---- my part, exactly the line in main.py
        x_hat, P = observer.update(data.sensordata, data.ctrl)
        err.append(x_hat - np.r_[data.qpos[arm.qadr], data.qvel[arm.dof]])     # compare with the true values
        # ---- stand-in controller (Yara's goes here, and it has to use x_hat, not data.qvel)
        qr, qdr, qddr = reference(data.time, q_home)
        data.ctrl[arm.act] = arm.ctrl_for_force(arm.inverse_dynamics(qr, qdr, qddr), qr, qdr)
        mujoco.mj_step(model, data)

    err = np.array(err)[int(1.0 / model.opt.timestep):]
    print("\n1) main.py loop: my estimate vs the true MuJoCo values (after the first 1 s)")
    print("   speed error per joint [rad/s]:", np.sqrt((err[:, 6:]**2).mean(0)))
    print("   angle error per joint [mrad] :", 1e3 * np.sqrt((err[:, :6]**2).mean(0)))

    # 2) does it give the same answers as step 4? run the same recorded trial through both
    from step4_Sameeha_Work.step3_measurement_and_ground_truth import run_trial
    from step4_Sameeha_Work.step4_observers import run_kf
    tr = run_trial(arm, q_home, T=3.0, seed=5)
    o2 = JointObserver(model, add_noise=False)
    X_obs = np.vstack([np.r_[tr["y"][0], np.zeros(6)]] +
                      [o2.step(tr["y"][k - 1], tr["u"][k - 1])[0] for k in range(1, len(tr["t"]))])
    X_ref, _ = run_kf(tr, Q_A, *linearise(arm, q_home))
    print("\n2) observer.py vs step 4 on the same data: biggest difference = %.1e" % np.abs(X_obs - X_ref).max())

    # 3) is it fast enough? main.py runs in real time with a 2 ms step
    import time
    t0 = time.perf_counter()
    for _ in range(2000):
        observer.update(data.sensordata, data.ctrl)
    us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"\n3) one update takes {us:.0f} us  ({100 * us / (1e6 * model.opt.timestep):.0f} % of the 2 ms step)")
