"""
step4_observers.py -> Task 2 (Sameeha): estimating the joint speeds (which aren't measured)

What's in here:
    finite_difference()   simple way: speed = (this angle - last angle) / dt   (the one to beat)
    luenberger_gain()     Luenberger observer gain from pole placement
    run_luenberger()      runs the Luenberger observer over a recorded trial
    kf_step()             one Kalman filter step (same as Week 7)
    run_kf()              runs the Kalman filter over a recorded trial
    W_O, Q_A              my tuned settings (found in step6_tuning.py)

Uses info from tutorials: Week 7 Part A: Luenberger observer, ct.place, duality; Week 7 Part B:
Kalman filter, kf_step, Qd / Rn, ct.lqe and Week 2: speed from encoder readings.
Criterion: C (observer design)

Run:  python step4_Sameeha_Work/step4_observers.py

How the observers work (prediction form, because of the main.py timing in step 3):
    at step k I have y[k-1] and u[k-1] and want x_hat[k]:

        x_hat[k] = Ad x_hat[k-1] + Bd u[k-1] + L ( y[k-1] - C x_hat[k-1] )

    i.e. predict with the model, then correct it with how wrong the encoder says it is.
    Error: e[k] = (Ad - L C) e[k-1], so I place the eigenvalues of (Ad - L C) like in Week 7
    (ct.place on Ad', C' then transpose). The Kalman filter is the same idea, it just picks L
    from the noise levels instead of me choosing poles.
"""

# sys + Path are imported so this file can find my other step files
import sys
from pathlib import Path

# numpy is imported for the matrix maths
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing the model functions from step 1
from step4_Sameeha_Work.step1_state_space_model import (                 # noqa: E402
    load_scene, linearise, double_integrator, make_Qd, NAMES, RESULTS)
# importing the trial recorder and the encoder noise from step 3
from step4_Sameeha_Work.step3_measurement_and_ground_truth import (      # noqa: E402
    run_trial, truth, Rn_for)

W_O = 49.0        # rad/s, how fast the Luenberger observer corrects   (tuned in step 6)
Q_A = 100.0       # Kalman filter process noise                       (tuned in step 6)


# ---------------------------------------------------------------- B.3 simple way
def finite_difference(tr):
    """Uses the same info as the observers: estimate of x[k] from y[k-1] and y[k-2]."""
    # Building the simple way to get speed: (new angle - old angle) / dt.
    # Dividing by dt = 0.002 makes the encoder noise ~700x bigger, which is why it's so noisy
    Y, dt = tr["y"], tr["t"][1] - tr["t"][0]
    q_hat = np.vstack([Y[:1], Y[:-1]])                           # latest reading y[k-1]
    # the first two speeds are 0 because there aren't two readings yet
    qd_hat = np.vstack([np.zeros((2, Y.shape[1])), np.diff(Y[:-1], axis=0) / dt])
    return np.hstack([q_hat, qd_hat])


# ---------------------------------------------------------------- B.4 Luenberger
def luenberger_gain(Ad, C, w_o, dt):
    """Places the poles of (Ad - L C). The fast servo poles are already fast so I leave them;
    the slow ones (s ~ -5) get moved to between -w_o and -1.5 w_o."""
    # python-control is imported for ct.place (pole placement from Week 7)
    import control as ct
    # the robot's own poles in s (sorted, most negative first)
    s_plant = np.sort(np.log(np.linalg.eigvals(Ad).astype(complex)).real / dt)    # most negative first
    # where I want the slow poles to go: between -1.5 w_o and -w_o
    target = -w_o * np.linspace(1.5, 1.0, len(s_plant))
    # keeping whichever is faster, so the fast servo poles don't get slowed down
    s_obs = np.minimum(s_plant, target)
    s_obs = s_obs - 1e-3 * np.arange(len(s_obs))                                # all slightly different
    # MAIN PART: the duality trick from Week 7. Placing the poles of (A' - C' K) is the same as (A - L C), so L = K'
    return ct.place(Ad.T, C.T, np.exp(s_obs * dt)).T


def run_luenberger(tr, L, Ad, Bd, x_star, u_star):
    Y, U = tr["y"], tr["u"]
    n = Y.shape[1]
    # C picks out the angles only (y = q)
    C = np.hstack([np.eye(n), np.zeros((n, n))])
    # starting guess: angle = first reading, speed = 0 (written as dx = x - x*, the change from home)
    dx = np.r_[Y[0] - x_star[:n], np.zeros(n)]
    Xh = np.zeros((len(Y), 2 * n)); Xh[0] = x_star + dx
    for k in range(1, len(Y)):
        # the observer equation: predict with the model (Ad dx + Bd du), then correct with L * (how wrong the encoder says I am)
        dx = Ad @ dx + Bd @ (U[k - 1] - u_star) + L @ (Y[k - 1] - x_star[:n] - C @ dx)
        Xh[k] = x_star + dx
    return Xh


# ---------------------------------------------------------------- B.5 Kalman filter
def kf_step(xh, P, u, y, A, B, C, Qd, Rn):
    """Week 7 kf_step but in prediction form: correct with y (last reading), then predict
    one step with u. Returns x_hat[k] and P[k]."""
    # S = how unsure I am about the reading (my uncertainty + the encoder noise)
    S = C @ P @ C.T + Rn
    # K = Kalman gain: how much to trust the encoder vs my model (worked out from P and Rn)
    K = P @ C.T @ np.linalg.inv(S)
    # CORRECT: moving the estimate towards the encoder reading by K * (how wrong I was)
    xh = xh + K @ (y - C @ xh)
    # the correction makes me more sure, so P gets smaller
    P = (np.eye(len(xh)) - K @ C) @ P
    # PREDICT: moving the estimate one step forward with the model. P grows by Qd because the model isn't perfect
    return A @ xh + B @ u, A @ P @ A.T + Qd


def run_kf(tr, q_a, Ad, Bd, x_star, u_star, Rn=None, arm=None, relin_every=None):
    """Runs the Kalman filter over a recorded trial. Returns estimates (N, 12) and P (N, 12, 12).
    relin_every = M re-linearises the model at the current estimate every M steps (step 7 test)."""
    Y, U = tr["y"], tr["u"]
    n, dt = Y.shape[1], tr["t"][1] - tr["t"][0]
    # using the encoder noise from step 3 unless a different Rn is given
    Rn = Rn_for() if Rn is None else Rn
    C = np.hstack([np.eye(n), np.zeros((n, n))])
    # making Qd from q_a (Q_A = 100 from tuning)
    Qd = make_Qd(q_a, dt, n)
    # starting guess: angles = first reading, speeds = 0
    xh = np.r_[Y[0], np.zeros(n)]
    # starting uncertainty: angles as unsure as the encoder, speeds unsure by 0.5 rad/s
    P = np.diag(np.r_[np.full(n, Rn[0, 0]), np.full(n, 0.5**2)])
    # arrays to save the estimate and P at every step
    Xh = np.zeros((len(Y), 2 * n)); Ph = np.zeros((len(Y), 2 * n, 2 * n))
    Xh[0], Ph[0] = xh, P
    # going through the trial one step at a time (the same thing observer.py does live)
    for k in range(1, len(Y)):
        if k == 1:                                       # first step same as observer.py
            dx, P = Ad @ (xh - x_star) + Bd @ (U[0] - u_star), Ad @ P @ Ad.T + Qd
        else:
            # one KF step using the last reading y[k-1] and the last command u[k-1] (the main.py timing)
            dx, P = kf_step(xh - x_star, P, U[k - 1] - u_star, Y[k - 1] - x_star[:n], Ad, Bd, C, Qd, Rn)
        xh = x_star + dx
        Xh[k], Ph[k] = xh, P
        # only for the step 7 test: rebuilding the model at the current estimated pose
        if relin_every and k % relin_every == 0:
            Ad, Bd, x_star, u_star = linearise(arm, xh[:n])
    return Xh, Ph


def rmse(e, axis=0):
    # RMSE = root mean square error = the average size of the error (what I report everywhere)
    return np.sqrt(np.mean(np.asarray(e)**2, axis=axis))


# ================================================================================ run
if __name__ == "__main__":
    # python-control is imported for ct.dlqe (the steady-state Kalman filter)
    import control as ct
    np.set_printoptions(precision=4, suppress=True, linewidth=130)

    # setting up: scene, model at home, C, Rn, and the same 10 s test trial as step 5
    model, arm, dt, q_home = load_scene()
    n = arm.n
    Ad, Bd, x_star, u_star = linearise(arm, q_home)
    _, _, C = double_integrator(dt)
    Rn = Rn_for()
    tr = run_trial(arm, q_home, T=10.0, seed=1)
    X = truth(tr)
    k0 = int(1.0 / dt)                                   # ignore the first 1 s (start-up)

    # prints the speed error per joint and the average angle error vs the truth
    def report(label, Xh):
        print(f"{label:24s} speed error per joint [rad/s]: {rmse(Xh[k0:, 6:] - X[k0:, 6:])}"
              f"   angle error {1e3 * rmse(Xh[k0:, :6] - X[k0:, :6]).mean():.3f} mrad")

    # ------------------------------------------------------------------ B.3 finite difference
    print("=" * 78)
    print("B.3 - finite difference: noise gets multiplied by sqrt(2)/dt (~707x at dt = 2 ms)")
    print("=" * 78)
    # running finite difference on the trial
    report("finite difference", finite_difference(tr))
    print(f"theory says sqrt(2)*sigma/dt = {np.sqrt(2 * Rn[0, 0]) / dt:.3f} rad/s")

    # ------------------------------------------------------------------ B.4 Luenberger
    print("\n" + "=" * 78)
    print(f"B.4 - Luenberger observer, w_o = {W_O} rad/s")
    print("=" * 78)
    # working out the Luenberger gain L with my tuned W_O
    L = luenberger_gain(Ad, C, W_O, dt)
    # checking where the error poles ended up compared with the robot's own poles
    s_err = np.sort(np.log(np.linalg.eigvals(Ad - L @ C).astype(complex)).real / dt)
    s_pl = np.sort(np.log(np.linalg.eigvals(Ad).astype(complex)).real / dt)
    print("robot poles  s [1/s]:", np.round(s_pl, 1))
    print("error poles  s [1/s]:", np.round(s_err, 1), " <- slow -5 ones moved, fast servo ones kept")
    report("Luenberger", run_luenberger(tr, L, Ad, Bd, x_star, u_star))

    # "model only" = L = 0, just running the model with no encoder correction
    report("model only (L = 0)", run_luenberger(tr, np.zeros_like(L), Ad, Bd, x_star, u_star))

    # ------------------------------------------------------------------ B.5 Kalman filter
    print("\n" + "=" * 78)
    print(f"B.5 - Kalman filter, Qd from q_a = {Q_A:g}, Rn from the encoder model")
    print("=" * 78)
    # MAIN PART: running my Kalman filter on the trial
    X_kf, P_kf = run_kf(tr, Q_A, Ad, Bd, x_star, u_star)
    report("KF (MuJoCo model)", X_kf)
    # the gain it settles to = the steady-state gain from ct.dlqe (discrete version of Week 7's ct.lqe)
    _, _, E = ct.dlqe(Ad, np.eye(12), C, make_Qd(Q_A, dt), Rn)
    # turning those poles into s so I can compare them with Luenberger's
    sE = np.sort(np.log(np.abs(E)) / dt)
    print("steady-state KF error poles s [1/s]:", np.round(sE, 1))
    print("Note: the steady-state KF is basically a Luenberger observer where the noise picked the"
          " poles instead of me.")

    # ------------------------------------------------------------------ B.6 without the model
    print("\n" + "=" * 78)
    print("B.6 - is my MuJoCo model worth it? Kalman filter with no model (constant speed)")
    print("=" * 78)
    # the no-model KF: double integrator with u = 0, so it only knows 'the speed stays about the same'
    Ad_di, Bd_di, _ = double_integrator(dt)
    tr0 = dict(tr, u=np.zeros_like(tr["u"]))
    X_cv, _ = run_kf(tr0, 2.0e3, Ad_di, Bd_di, np.zeros(12), np.zeros(n))
    report("KF (const. velocity)", X_cv)