"""
step1_state_space_model.py -> Task 2 (Sameeha): state-space model of the UR3e

What's in here:
    Arm                  finds the 6 arm joints in the team scene + the actuator formula
    load_scene()         loads the team scene from environment.py (same world as main.py)
    double_integrator()  textbook model (qdd = u for each joint)
    linearise()          the model I actually use: MuJoCo linearised around a pose
    make_Qd()            process noise for the Kalman filter (used in step 4)

Uses info from tutorials: Week 5: state space, linearisation; Week 6: exact ZOH discretisation and Week 7: C.2 Jacobian linearisation.
Criterion: A (Modelling)

Run:  python step4_Sameeha_Work/step1_state_space_model.py      (graphs go in results/)

My state, input and output (same in every step):
    x = [q; qd]   12 states -> 6 joint angles q [rad] + 6 joint speeds qd [rad/s]
    u             6 inputs  -> the motor commands (for this ur3e.xml they're angle setpoints)
    y = q + v     6 outputs -> encoders only measure the angles, NOT the speeds
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import step1_Main.environment as env                                     # noqa: E402

RESULTS = HERE / "results"
UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
STATE_NAMES = [f"q_{n}" for n in NAMES] + [f"qd_{n}" for n in NAMES]


# ================================================================ finding the arm in the scene
class Arm:
    """The 6 UR joints inside the scene.

    The scene also has the red cube (it has a free joint, so it adds extra entries to qpos/qvel),
    so I look every joint up by name instead of assuming it's index 0-5.
    """

    def __init__(self, model):
        self.m = model
        self.jid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in UR_JOINTS])
        if np.any(self.jid < 0):
            raise ValueError(f"model is missing UR joints: {UR_JOINTS}")
        self.qadr = model.jnt_qposadr[self.jid]          # where each joint is in qpos
        self.dof = model.jnt_dofadr[self.jid]            # where each joint is in qvel
        self.act = np.array([np.where((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                      & (model.actuator_trnid[:, 0] == j))[0][0] for j in self.jid])
        # MuJoCo actuator formula: force = gain*u + b0 + b1*q + b2*qd
        self.gain = model.actuator_gainprm[self.act, 0]
        self.bias = model.actuator_biasprm[self.act, :3]
        self.gear = model.actuator_gear[self.act, 0]
        self.act_type = "position" if np.all(self.bias[:, 1] < 0) else "motor"
        self.kp, self.kv = -self.bias[:, 1], -self.bias[:, 2]
        self.ctrl_lo, self.ctrl_hi = model.actuator_ctrlrange[self.act].T
        self.n = len(self.jid)
        self._d = mujoco.MjData(model)                   # spare MjData I reuse for calculations

    def ctrl_for_force(self, tau, q, qd):
        """Which command u gives joint torque tau (rearranged actuator formula)."""
        b, g = self.bias, self.gear
        return (tau / g - b[:, 0] - b[:, 1] * g * q - b[:, 2] * g * qd) / self.gain

    def inverse_dynamics(self, q, qd, qdd):
        """Torque needed for a motion: tau = M(q) qdd + C(q, qd) qd + g(q)."""
        d = self._d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.qadr], d.qvel[self.dof], d.qacc[self.dof] = q, qd, qdd
        mujoco.mj_inverse(self.m, d)
        return d.qfrc_inverse[self.dof].copy()

    def hold_ctrl(self, q0):
        """Command that holds the arm still at q0 (has to push against gravity)."""
        z = np.zeros(self.n)
        return self.ctrl_for_force(self.inverse_dynamics(q0, z, z), q0, z)

    def mass_matrix(self, q):
        """6x6 inertia matrix M(q) at pose q."""
        d = self._d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, d)
        M = np.zeros((self.m.nv, self.m.nv))
        try:
            mujoco.mj_fullM(self.m, d, M)                # MuJoCo 3.17
        except TypeError:
            mujoco.mj_fullM(self.m, M, d.qM)             # older MuJoCo (3.16)
        return M[np.ix_(self.dof, self.dof)]


def load_scene():
    """Loads the team scene. Gives back model, arm, dt and the home pose."""
    model = env.build_scene()
    arm = Arm(model)
    q_home = model.key_qpos[0][arm.qadr].copy()        # keyframe 0 = "home"
    return model, arm, model.opt.timestep, q_home


# ================================================================ model 1: textbook
def double_integrator(dt, n=6):
    """If the controller cancels gravity + inertia, each joint is just qdd = a (a double integrator).
    Discretised exactly (ZOH, Week 6). x = [q; qd], u = a, y = q."""
    I, Z = np.eye(n), np.zeros((n, n))
    Ad = np.block([[I, dt * I], [Z, I]])
    Bd = np.vstack([0.5 * dt**2 * I, dt * I])
    C = np.hstack([I, Z])
    return Ad, Bd, C


# ================================================================ model 2: from MuJoCo
def linearise(arm, q0, n_sub=1, eps=1e-6):
    """Linearise the actual MuJoCo robot around the pose q0 (standing still).

        dx[k+1] = Ad dx[k] + Bd du[k]      where dx = x - x*, du = u - u*

    mjd_transitionFD gives the Jacobians of one physics step (Week 5/7 linearisation, but MuJoCo
    does the maths). n_sub is only needed if the observer runs slower than the physics.
    Returns Ad (12x12), Bd (12x6), x_star, u_star."""
    m, n = arm.m, arm.n
    d = mujoco.MjData(m)
    d.qpos[arm.qadr] = q0
    d.ctrl[arm.act] = arm.hold_ctrl(q0)
    mujoco.mj_forward(m, d)
    nx = 2 * m.nv + m.na
    A = np.zeros((nx, nx)); B = np.zeros((nx, m.nu))
    mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None)
    idx = np.r_[arm.dof, m.nv + arm.dof]                 # keep the arm's 12 states, drop the cube's
    A, B = A[np.ix_(idx, idx)], B[np.ix_(idx, arm.act)]
    Bd, Ak = np.zeros_like(B), np.eye(2 * n)
    for _ in range(n_sub):
        Bd += Ak @ B
        Ak = A @ Ak
    return Ak, Bd, np.r_[q0, np.zeros(n)], d.ctrl[arm.act].copy()


def make_Qd(q_a, dt, n=6):
    """Process noise Qd: assumes random joint accelerations of strength q_a.
        Qd = q_a [[dt^4/4 I, dt^3/2 I], [dt^3/2 I, dt^2 I]]"""
    G = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    return q_a * np.kron(G, np.eye(n))


def stamp(name, M):
    print(f"{name} =")
    print(np.atleast_2d(M))


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["figure.figsize"] = (9, 3.2)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    model, arm, dt, q_home = load_scene()
    n = arm.n

    # ------------------------------------------------------------------ Part 0: the scene
    print("=" * 78)
    print("Part 0 - the scene")
    print("=" * 78)
    print("robot xml :", env.ROBOT_XML)
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}   (6 arm joints + the cube's free joint)")
    print("timestep  :", dt, "s   (main.py runs the observer every step, so my model uses this dt)")
    print("actuators :", arm.act_type, "servos   kp =", arm.kp, "  kv =", arm.kv)
    print("q_home    :", q_home)

    # ------------------------------------------------------------------ A.2 M(q) changes with pose
    # Week 5: a linear model is only exact close to where you linearise it.
    print("\n" + "=" * 78)
    print("A.2 - the robot is nonlinear: M(q) changes with the pose")
    print("=" * 78)
    stamp("M(q_home) [kg m^2]", arm.mass_matrix(q_home))
    poses = {"home": q_home,
             "arm stretched": np.zeros(n),
             "elbow folded": q_home + np.array([0, 0.6, 0.9, 0, 0, 0])}
    print("\n  pose              M11      M22      M33    cond(M)")
    for k, qp in poses.items():
        Mq = arm.mass_matrix(qp)
        print(f"  {k:15s} {Mq[0, 0]:8.4f} {Mq[1, 1]:8.4f} {Mq[2, 2]:8.4f} {np.linalg.cond(Mq):9.1f}")
    print("-> base inertia changes a lot with pose, so one linear model is only exact near q0")

    # ------------------------------------------------------------------ A.3 textbook model
    # Week 5/6: xdot = [[0 I],[0 0]] x + [[0],[I]] a,  y = [I 0] x, then discretise exactly.
    print("\n" + "=" * 78)
    print("A.3 - model 1: double integrator (textbook)")
    print("=" * 78)
    Ad_di, Bd_di, C = double_integrator(dt)
    stamp("Ad (joint 1)", Ad_di[np.ix_([0, 6], [0, 6])])
    stamp("Bd (joint 1)", Bd_di[np.ix_([0, 6], [0])])
    print("all eigenvalues of Ad = 1 (only just stable) ->",
          np.allclose(np.linalg.eigvals(Ad_di), 1))

    # ------------------------------------------------------------------ A.4 MuJoCo model
    print("\n" + "=" * 78)
    print("A.4 - model 2: linearise the MuJoCo UR3e (mjd_transitionFD)")
    print("=" * 78)
    Ad, Bd, x_star, u_star = linearise(arm, q_home)
    print("u* (command to hold still) =", u_star)
    print("u* - q_home        =", u_star - q_home, " <- extra setpoint the servos need to fight gravity")
    stamp("Ad (shoulder lift)", Ad[np.ix_([1, 7], [1, 7])])
    stamp("Bd (shoulder lift)", Bd[np.ix_([1, 7], [1])])

    # check: hold u* for 1 s, the arm shouldn't move at all
    d = mujoco.MjData(model)
    env.reset_home(model, d)
    d.ctrl[arm.act] = u_star
    for _ in range(int(1.0 / dt)):
        mujoco.mj_step(model, d)
    print("\nequilibrium check, 1 s holding u*:  max |q - q0| = %.2e rad,  max |qd| = %.2e rad/s"
          % (np.abs(d.qpos[arm.qadr] - q_home).max(), np.abs(d.qvel[arm.dof]).max()))

    # poles: convert z to s with s = ln(z)/dt
    z = np.linalg.eigvals(Ad)
    s = np.log(z.astype(complex)) / dt
    order = np.argsort(s.real)
    print("\n   |z|        s = ln(z)/dt  [1/s]")
    for zi, si in zip(z[order], s[order]):
        print(f"  {abs(zi):.5f}   {si.real:9.2f} {si.imag:+9.2f}j")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(np.cos(th), np.sin(th), "k", lw=0.8)
    ax[0].plot(z.real, z.imag, "bx", ms=8, label="MuJoCo model")
    ax[0].plot(1, 0, "ro", mfc="none", ms=10, label="double integrator (x12)")
    ax[0].set_aspect("equal")
    ax[0].set_title("discrete poles z"); ax[0].legend(fontsize=8, loc="lower left")
    ax[1].plot(s.real, s.imag, "bx", ms=8)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Re s [1/s]"); ax[1].set_ylabel("Im s [rad/s]")
    ax[1].set_title("same poles in continuous time, s = ln(z)/dt")
    plt.tight_layout(); fig.savefig(RESULTS / "step1_poles.png", dpi=150)

    # Note: the servos make each joint act like a stiff spring with lots of damping.
    # Per joint: J qdd + kv qd + kp q = kp u -> one slow pole at about -kp/kv = -5 and one fast one.
    # kp/kv = 5 for every joint in ur3e.xml, that's why 6 poles sit together at s = -5.
    # This is the plant the observer has to track, so steps 3-7 use this model, not the double integrator.

    # check the model: small step in u, compare the linear model with MuJoCo
    du = 0.05 * np.ones(n)
    K = int(1.0 / dt)
    d = mujoco.MjData(model)
    d.qpos[arm.qadr] = q_home
    mujoco.mj_forward(model, d)
    X_nl = np.zeros((K + 1, 2 * n)); X_lin = np.zeros_like(X_nl)
    X_nl[0] = X_lin[0] = x_star
    dx = np.zeros(2 * n)
    for k in range(K):
        d.ctrl[arm.act] = u_star + du
        mujoco.mj_step(model, d)
        X_nl[k + 1] = np.r_[d.qpos[arm.qadr], d.qvel[arm.dof]]
        dx = Ad @ dx + Bd @ du
        X_lin[k + 1] = x_star + dx
    tk = np.arange(K + 1) * dt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for j in range(n):
        ax[0].plot(tk, X_nl[:, j] - q_home[j], f"C{j}", lw=2, label=NAMES[j])
        ax[0].plot(tk, X_lin[:, j] - q_home[j], f"C{j}--", lw=1)
        ax[1].plot(tk, X_nl[:, 6 + j], f"C{j}", lw=2)
        ax[1].plot(tk, X_lin[:, 6 + j], f"C{j}--", lw=1)
    ax[0].set_title("q - q0: MuJoCo (solid) vs my model (dashed)"); ax[0].set_ylabel("rad")
    ax[1].set_title("qd: MuJoCo (solid) vs my model (dashed)"); ax[1].set_ylabel("rad/s")
    for a in ax:
        a.set_xlabel("t [s]")
    ax[0].legend(fontsize=7, ncol=3)
    plt.tight_layout(); fig.savefig(RESULTS / "step1_model_validation.png", dpi=150)
    fit = 100 * (1 - np.linalg.norm(X_nl - X_lin) / np.linalg.norm(X_nl - X_nl.mean(0)))
    print(f"\nhow well my linear model matches MuJoCo (1 s step): {fit:.1f} %")

    print("\nfigures saved to", RESULTS)
    plt.show()
