"""
step3_measurement_and_ground_truth.py -> Task 2 (Sameeha): realistic encoder readings + the true motion

What's in here:
    encoder()        noisy encoder reading: true angle + noise, rounded to the encoder's steps
    Rn_for()         measurement noise matrix  Rn = (sigma^2 + Delta^2 / 12) I
    reference()      smooth test motion around home (q, qd, qdd)
    run_trial()      moves the robot in MuJoCo and records the true angles/speeds, the commands
                     and the encoder readings -> this is the "answer key" I grade against

Uses info from tutorials: Week 2: encoders as quantisers (z = h(x) + b + v) and Week 7: truth vs estimate.
Criterion: B/C (noise handling + ground truth for estimation)

Run:  python step4_Sameeha_Work/step3_measurement_and_ground_truth.py

Timing note (matches main.py):
    MuJoCo fills data.sensordata INSIDE mj_step, before it moves the robot. So after mj_step
    the sensor reading is from the start of that step, one step (2 ms) behind data.qpos.
    In main.py the observer gets y[k-1] and u[k-1] and has to estimate x[k].
    So every estimator in steps 4-7 only uses y[0..k-1] and u[0..k-1] to estimate x[k].
"""

# sys + Path are imported so this file can find my other step files
import sys
from pathlib import Path

# numpy is imported for the random noise and arrays
import numpy as np
# mujoco is imported to actually move the robot and record the true motion
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing the scene loader from step 1
from step4_Sameeha_Work.step1_state_space_model import load_scene, env, NAMES, RESULTS   # noqa: E402

# ---------------------------------------------------------------- B.1 encoder model
SIGMA_ENC = 1e-3                   # rad, encoder noise. On purpose worse than a real UR encoder so
                                   # the methods actually look different (step 6 tests other values)
DELTA_ENC = 2 * np.pi / 2**14      # rad, step size of a 14-bit encoder


def encoder(q, rng, sigma=SIGMA_ENC, delta=DELTA_ENC):
    """y = q + noise, then rounded to the nearest encoder step (Week 2: rounding adds noise Delta^2/12)."""
    # adding random noise with std sigma to the true angle (like a real sensor)
    y = q + rng.normal(0.0, sigma, np.shape(q))
    # rounding to the nearest encoder step (a 14-bit encoder can only show 16384 positions per turn)
    return np.round(y / delta) * delta if delta > 0 else y


def Rn_for(sigma=SIGMA_ENC, delta=DELTA_ENC, n=6):
    # Building Rn, the measurement noise for the Kalman filter: random noise^2 + rounding noise (Delta^2/12).
    # This tells the KF how much it can trust each encoder reading
    return (sigma**2 + delta**2 / 12) * np.eye(n)


# ---------------------------------------------------------------- B.2 test motion
AMP = np.array([0.40, 0.30, 0.30, 0.40, 0.40, 0.50])         # rad, roughly the size of a pick motion
FREQ = np.array([[0.13, 0.31], [0.17, 0.43], [0.21, 0.37],
                 [0.19, 0.53], [0.23, 0.47], [0.29, 0.61]])   # Hz, two sine waves per joint


def _q_ref(t, q0, amp):
    env = 0.5 * (1 - np.cos(np.pi * min(max(t, 0.0), 1.0)))  # fades in smoothly over the first 1 s
    # two sine waves per joint so the motion isn't too simple or repeating
    wave = 0.6 * np.sin(2 * np.pi * FREQ[:, 0] * t) + 0.4 * np.sin(2 * np.pi * FREQ[:, 1] * t)
    return q0 + env * amp * wave


def reference(t, q0, amp=AMP):
    """Gives q_ref, qd_ref, qdd_ref (speeds/accelerations by finite differences of the smooth curve)."""
    h = 1e-4
    # speed and acceleration of the test motion, worked out with tiny finite differences (h = 0.0001 s)
    qm, qc, qp = _q_ref(t - h, q0, amp), _q_ref(t, q0, amp), _q_ref(t + h, q0, amp)
    return qc, (qp - qm) / (2 * h), (qp - 2 * qc + qm) / h**2


def run_trial(arm, q0, T=10.0, seed=0, amp=AMP, sigma=SIGMA_ENC, model=None, disturbance=None):
    """Moves the UR3e along reference() and records everything.

    This is just a TEST setup: the commands come straight from the reference (the servos do
    their own feedback). The real controller is Yara's and it has to use x_hat from observer.py.

    model        robot to simulate (default: arm.m). Pass a changed copy to test a wrong model.
    disturbance  f(t) -> 6 joint torques [N m] the observer doesn't know about.
    Returns a dict: t, q, qd (true), u (commands), y (encoder readings), qref."""
    m = arm.m if model is None else model
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    # starting the robot at the home keyframe, then moving it to pose q0
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[arm.qadr] = q0
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    # number of steps, e.g. 10 s / 0.002 s = 5000
    N = int(round(T / dt))
    # empty arrays to save everything in, one row per time step
    out = {k: np.zeros((N, arm.n)) for k in ["q", "qd", "u", "y", "qref"]}
    out["t"] = np.arange(N) * dt
    # This is the loop that moves the robot one step at a time and records everything
    for k in range(N):
        t = k * dt
        # the TRUE angles and speeds from MuJoCo (this is my answer key)
        q, qd = d.qpos[arm.qadr].copy(), d.qvel[arm.dof].copy()
        # where the test motion wants the arm to be right now
        qr, qdr, qddr = reference(t, q0, amp)
        tau_ff = arm.inverse_dynamics(qr, qdr, qddr)                 # uses the normal model (no payload)
        # turning that torque into the servo command u (the observer gets told u)
        if arm.act_type == "position":
            u = arm.ctrl_for_force(tau_ff, qr, qdr)                  # setpoint + extra for gravity/inertia
        else:                                                        # (if motors were torque motors: add PD)
            u = arm.ctrl_for_force(tau_ff + 400 * (qr - q) + 40 * (qdr - qd), q, qd)
        # keeping u inside the motor limits
        u = np.clip(u, arm.ctrl_lo, arm.ctrl_hi)
        d.ctrl[arm.act] = u
        d.qfrc_applied[:] = 0
        # adding a push or load that the observer doesn't know about (only for the robustness tests)
        if disturbance is not None:
            d.qfrc_applied[arm.dof] = disturbance(t)
        # saving this step: true q and qd, the command u, the NOISY encoder reading y, and the reference
        for key, val in zip(["q", "qd", "u", "y", "qref"], [q, qd, u, encoder(q, rng, sigma), qr]):
            out[key][k] = val
        # moving the physics forward one step (2 ms)
        mujoco.mj_step(m, d)
    return out


def truth(tr):
    # joining the true angles and speeds into the full 12-state truth x = [q; qd]
    return np.hstack([tr["q"], tr["qd"]])


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    model, arm, dt, q_home = load_scene()

    # ------------------------------------------------------------------ B.1
    print("=" * 78)
    print("B.1 - encoder model")
    print("=" * 78)
    Rn = Rn_for()
    print(f"sigma_enc = {SIGMA_ENC:.1e} rad   encoder step = {DELTA_ENC:.2e} rad")
    print(f"Rn = {Rn[0, 0]:.3e} I   ->  total noise std = {np.sqrt(Rn[0, 0]):.2e} rad")
    # check: taking 20000 fake readings and making sure the noise std matches Rn
    rng = np.random.default_rng(0)
    e = np.array([encoder(q_home, rng) - q_home for _ in range(20000)])
    print(f"check: std of 20000 readings = {e.std(0).mean():.3e} rad  (should match the line above)")

    # timing check: after mj_step, sensordata is the q from BEFORE the step
    # importing observer.py here just to add the encoder sensors for this timing test
    import step4_Sameeha_Work.observer as obs
    spec = mujoco.MjSpec.from_file(str(env.ROBOT_XML))
    obs.add_encoder_sensors(spec)
    m_s = spec.compile()
    d = mujoco.MjData(m_s)
    mujoco.mj_resetDataKeyframe(m_s, d, 0)
    d.ctrl[0] += 0.5                                     # make the base turn
    for _ in range(50):
        mujoco.mj_step(m_s, d)
    q_before = d.qpos[:6].copy()
    # one more step, then comparing sensordata with the angle now and the angle from before the step
    mujoco.mj_step(m_s, d)
    print("\nMuJoCo sensor timing after mj_step:")
    print(f"  |sensordata - qpos now|         = {np.abs(d.sensordata - d.qpos[:6]).max():.2e} rad")
    print(f"  |sensordata - qpos before step| = {np.abs(d.sensordata - q_before).max():.2e} rad   <- one step old")

    # ------------------------------------------------------------------ B.2
    print("\n" + "=" * 78)
    print("B.2 - ground truth: move the UR3e along the test motion")
    print("=" * 78)
    # recording the 10 s test motion with noise seed 1 (the same one steps 4 and 5 use)
    tr = run_trial(arm, q_home, T=10.0, seed=1)
    print(f"{len(tr['t'])} samples at {dt} s")
    print("fastest speed per joint [rad/s]:", np.abs(tr["qd"]).max(0))
    print("how far it lags the reference, max [rad]:", np.abs(tr["q"] - tr["qref"]).max(0))

    # check the arm doesn't hit the table or the cube during the test motion
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    zmin, ncon_max = np.inf, 0
    # going through the motion every 10 steps and checking the tool height and arm contacts
    for q in tr["q"][::10]:
        d.qpos[arm.qadr] = q
        mujoco.mj_forward(model, d)
        zmin = min(zmin, d.site_xpos[site][2])
        ncon_max = max(ncon_max, sum(1 for c in d.contact[:d.ncon]
                                     if model.geom_bodyid[c.geom1] in model.jnt_bodyid[arm.jid]
                                     or model.geom_bodyid[c.geom2] in model.jnt_bodyid[arm.jid]))
    print(f"lowest tool height = {zmin:.3f} m   arm collisions during the motion = {ncon_max}")

    # graph: reference (dashed) vs what MuJoCo actually did, for each joint -> step3_trajectory.png
    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    for j, a in enumerate(ax.flat):
        a.plot(tr["t"], tr["qref"][:, j], "k--", lw=1, label="reference")
        a.plot(tr["t"], tr["q"][:, j], "C0", lw=1.5, label="MuJoCo truth")
        a.set_title(NAMES[j]); a.set_ylabel("q [rad]")
    for a in ax[1]:
        a.set_xlabel("t [s]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("B.2 test motion that every estimator gets graded on")
    plt.tight_layout(); fig.savefig(RESULTS / "step3_trajectory.png", dpi=150)

    # graph: 0.2 s zoomed in, true angle vs the noisy stepped encoder reading -> step3_encoder_noise.png
    w = slice(int(3 / dt), int(3.2 / dt))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tr["t"][w], tr["q"][w, 0], "k", lw=2, label="true q (pan)")
    ax.step(tr["t"][w], tr["y"][w, 0], "C3", where="post", lw=0.8, label="encoder reading y")
    ax.set_xlabel("t [s]"); ax.set_ylabel("rad"); ax.legend(fontsize=8)
    ax.set_title("B.1 encoder noise + rounding (zoomed to 0.2 s)")
    plt.tight_layout(); fig.savefig(RESULTS / "step3_encoder_noise.png", dpi=150)

    print("\nfigures saved to", RESULTS)
    plt.show()