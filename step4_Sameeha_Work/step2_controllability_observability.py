"""
step2_controllability_observability.py -> Task 2 (Sameeha): can the motors control it, can the encoders see it?

What's in here:
    rank_ctrb_obsv()       rank tests (yes/no answer)
    gramians()             Gramians (HOW controllable / observable, small eigenvalue = hard)
    pbh_detectable_d()     PBH test: any unstable mode has to show up in y
    pbh_stabilisable_d()   PBH test: any unstable mode has to be reachable from u

Uses info from tutorials: Week 5: ctrb / obsv rank tests and Week 8: Gramians, PBH,
stabilisable / detectable. Changed to discrete time, so "unstable" means |lambda| >= 1.
Criterion: A (controllability and observability)

Run:  python step4_Sameeha_Work/step2_controllability_observability.py
"""

# sys + Path are imported so this file can find my other step files
import sys
from pathlib import Path

# numpy is imported for the rank tests and eigenvalues
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
# importing the scene and both models from step 1 so I don't have to make them again
from step4_Sameeha_Work.step1_state_space_model import (                 # noqa: E402
    load_scene, double_integrator, linearise, STATE_NAMES, RESULTS)


def ctrb(A, B):
    """[B, AB, ..., A^(n-1) B]  (same as ct.ctrb from Week 5)."""
    # Building the controllability matrix: B, AB, A^2 B ... side by side.
    # If it has rank 12, the motors can push the arm into any state
    n = A.shape[0]
    blocks, Ak = [], np.eye(n)
    for _ in range(n):
        blocks.append(Ak @ B)
        Ak = A @ Ak
    return np.hstack(blocks)


def obsv(A, C):
    """[C; CA; ...; C A^(n-1)]  (same as ct.obsv from Week 5)."""
    # observability is the same thing flipped (duality, Week 7), so I just reuse ctrb with A' and C'.
    # If it has rank 12, I can work out all 12 states from the encoder readings
    return ctrb(A.T, C.T).T


def rank_ctrb_obsv(A, B, C, tol=1e-9):
    # matrix_rank counts how many independent directions there are (12 = full rank = yes)
    return (np.linalg.matrix_rank(ctrb(A, B), tol=tol),
            np.linalg.matrix_rank(obsv(A, C), tol=tol))


def gramians(A, B, C, N):
    """Gramians added up over N steps.
    I use a finite horizon because the double integrator never settles, so the normal one doesn't exist."""
    # This is to calculate the Gramians by adding up each step's contribution over N steps
    Wc = np.zeros_like(A); Wo = np.zeros_like(A); Ak = np.eye(A.shape[0])
    for _ in range(N):
        # Wc = how much the input u can move each state (big = easy to control)
        Wc += Ak @ B @ B.T @ Ak.T
        # Wo = how much each state shows up in y (big = easy to see with the encoders)
        Wo += Ak.T @ C.T @ C @ Ak
        Ak = A @ Ak
    return Wc, Wo


def pbh_detectable_d(A, C, tol=1e-8):
    """True if every mode with |lambda| >= 1 keeps rank[A - lambda I; C] = n."""
    # PBH test: for every pole that doesn't die out by itself (|lambda| >= 1), check the encoders can see it.
    # If one can't be seen AND doesn't die out, no observer could ever estimate it
    n = A.shape[0]
    bad = [lam for lam in np.linalg.eigvals(A)
           if abs(lam) >= 1 - tol and np.linalg.matrix_rank(np.vstack([A - lam * np.eye(n), C]), tol=1e-6) < n]
    return len(bad) == 0, bad


def pbh_stabilisable_d(A, B, tol=1e-8):
    """True if every mode with |lambda| >= 1 keeps rank[A - lambda I, B] = n."""
    # same idea, but checking the motors can reach every pole that doesn't die out by itself
    n = A.shape[0]
    bad = [lam for lam in np.linalg.eigvals(A)
           if abs(lam) >= 1 - tol and np.linalg.matrix_rank(np.hstack([A - lam * np.eye(n), B]), tol=1e-6) < n]
    return len(bad) == 0, bad


# ================================================================================ run
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    plt.rcParams["axes.grid"] = True
    RESULTS.mkdir(exist_ok=True)

    # loading the scene and making both models (textbook + MuJoCo) so I can test them side by side
    model, arm, dt, q_home = load_scene()
    n = arm.n
    Ad_di, Bd_di, C = double_integrator(dt)
    Ad, Bd, _, _ = linearise(arm, q_home)
    models = [("double integrator", Ad_di, Bd_di), ("MuJoCo model", Ad, Bd)]

    # ------------------------------------------------------------------ rank tests (Week 5)
    print("=" * 78)
    print("A.5 - rank tests: n = 12, y = q (6 encoders), u = 6 motors")
    print("=" * 78)
    # rank tests on both models with y = q (the 6 encoders)
    for label, A_, B_ in models:
        rc, ro = rank_ctrb_obsv(A_, B_, C)
        print(f"{label:20s} rank ctrb = {rc:2d}   rank obsv = {ro:2d}   ->  "
              f"{'controllable' if rc == 12 else 'NOT controllable'}, {'observable' if ro == 12 else 'NOT observable'}")

    # ------------------------------------------------------------------ Gramians (Week 8)
    # smallest eigenvalue = the hardest direction to control/see; its eigenvector says which state
    print("\n" + "=" * 78)
    print("Gramians over 1 s: how controllable / observable it is, not just yes or no")
    print("=" * 78)
    # 1 s worth of steps for the Gramian
    N = int(1.0 / dt)
    gram_eigs = {}
    for label, A_, B_ in models:
        Wc, Wo = gramians(A_, B_, C, N)
        for wname, W in [("Wc", Wc), ("Wo", Wo)]:
            # eigh gives the eigenvalues sorted; the smallest one = the hardest direction to control/see
            ev, V = np.linalg.eigh(W)
            gram_eigs[(label, wname)] = ev
            # finding which state that weakest direction mostly points at
            weak = STATE_NAMES[np.argmax(np.abs(V[:, 0]))]
            print(f"{label:18s} {wname}: lambda_min = {ev[0]:.3e}   lambda_max = {ev[-1]:.3e}   "
                  f"cond = {ev[-1] / ev[0]:.2e}   weakest ~ {weak}")

    # graph of all 12 Gramian eigenvalues (log scale) for both models -> step2_gramians.png
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, wname in enumerate(["Wc", "Wo"]):
        for label, _, _ in models:
            ax[i].semilogy(np.arange(1, 13), gram_eigs[(label, wname)], "o-", label=label)
        ax[i].set_xlabel("eigenvalue number (sorted)"); ax[i].set_title(f"{wname} eigenvalues, 1 s horizon")
    ax[0].legend(fontsize=8)
    plt.tight_layout(); fig.savefig(RESULTS / "step2_gramians.png", dpi=150)

    # Note: the weakest observable directions are the speeds. We never measure speed, it only shows
    # up through how the angle changes over time. That's why estimating qd is a trade-off between
    # noise and lag (tuned in step 6).

    # ------------------------------------------------------------------ PBH + losing sensors (Week 8)
    print("\n" + "=" * 78)
    print("PBH test and 'what if the sensors were different'")
    print("=" * 78)
    # making different C matrices for 'what if the sensors were different'
    Z6, I6 = np.zeros((n, n)), np.eye(n)
    C_vel_only = np.hstack([Z6, I6])                   # only speed sensors, no encoders
    C_wrist3_lost = np.delete(C, 5, axis=0)            # wrist 3 encoder broken
    C_base3 = C[:3]                                    # only the 3 base joints measured
    print(f"{'output y':26s} {'model':18s} rank obsv  detectable   lambda_min(Wo)")
    # for each sensor setup and each model: rank, PBH detectable, and the smallest Gramian eigenvalue
    for cname, Cx in [("encoders, y = q", C), ("speed sensors only", C_vel_only),
                      ("wrist 3 encoder lost", C_wrist3_lost), ("base 3 joints only", C_base3)]:
        for label, A_, B_ in models:
            r = np.linalg.matrix_rank(obsv(A_, Cx), tol=1e-9)
            det_ok, _ = pbh_detectable_d(A_, Cx)
            wo_min = np.linalg.eigvalsh(gramians(A_, B_, Cx, N)[1])[0]
            print(f"{cname:26s} {label:18s} {r:5d}/12   {str(det_ok):10s}   {wo_min:.2e}")
    # checking the motors can stabilise everything (Yara's controller needs this)
    for label, A_, B_ in models:
        print(f"stabilisable from u ({label}): {pbh_stabilisable_d(A_, B_)[0]}")

    # printing what I worked out from the numbers above
    print("""
What I found
  1. With encoders (y = q) both models are controllable, observable, stabilisable and detectable.
     So by the separation principle (Week 8) the observer (me) and the controller (Yara) can be
     designed separately.
  2. Why lambda_min(Wo) is small even with encoders: the weakest direction is one of the FAST servo
     modes (s ~ -200 to -500, see step 1). A speed kick in that mode dies out in ~10 ms and barely
     moves q, so the encoders hardly see it. That's fine because it dies out by itself (that's what
     detectable means), but the observer can only really improve the slow s ~ -5 modes.
  3. Speed sensors only: the double integrator can't work out the actual angle (a constant offset
     in q gives no speed). The MuJoCo model still says rank 12, but only because the servo spring
     kp(u - q) links angle to acceleration. So that info comes from the actuator model, and if kp
     was wrong the angle estimate would be too. Encoders measure q directly, which is better.
  4. One encoder lost: the double integrator loses that joint completely. The MuJoCo model still
     says rank 12 (the joints are linked through inertia), but lambda_min(Wo) drops by about
     7 orders of magnitude, so that joint would be estimated really badly. The rank test says yes
     but the Gramian says it's not usable. That's why I check Gramians and not just rank (Week 8).
""")

    print("figures saved to", RESULTS)
    plt.show()