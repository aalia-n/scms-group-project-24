"""
step7_evaluate.py -- Task 1 (Aalia): quantitative evaluation of the perception pipeline.

This is the ONLY file that reads the object's true position, and only to score the estimates.

1. Randomised trials: N_TRIALS random object positions and yaws. At each, N_FRAMES noisy
   frames are measured and fused by the TargetFilter. Reports detection rate, single-frame
   and filtered errors, and NEES (is the reported covariance honest about the real error?).
2. Noise sweep: the sensor noise is scaled from 0x to 10x to show how error grows with it.

Results: printed table and results/evaluation.png.

Run (from the repo root):  python step3_Aalia_Work/step7_evaluate.py
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

import step1_Main.environment as env
import step3_Aalia_Work.step1_camera as cm
from step3_Aalia_Work.step6_perception import ObjectPerception, TargetFilter

N_TRIALS = 100
N_FRAMES = 10
NOISE_SCALES = [0, 1, 2, 5, 10]
SWEEP_TRIALS = 30
CHI2_3_95 = 7.815          # 95% bound of chi-squared with 3 degrees of freedom (Assessment 3)
BASE_NOISE = (cm.SIGMA_D, cm.SIGMA_RGB, cm.P_DROPOUT)


def set_noise_scale(k):
    """Scale the camera noise model by k (0 = noise-free, 1 = nominal)."""
    cm.SIGMA_D = tuple(k * s for s in BASE_NOISE[0])
    cm.SIGMA_RGB = k * BASE_NOISE[1]
    cm.P_DROPOUT = k * BASE_NOISE[2]


def nees(e, P):
    """Normalised estimation error squared, e^T P^-1 e. A consistent estimate averages 3 (3 states)."""
    return float(e @ np.linalg.solve(P, e))


def run_trials(model, data, perception, rng, n_trials, n_frames):
    """One dict per trial (None if the object was never detected)."""
    trials = []
    for i in range(n_trials):
        print(f"  trial {i + 1}/{n_trials}", end="\r")
        env.randomise_object(model, data, rng)
        p_true = data.body(env.OBJ_NAME).xpos.copy()          # ground truth: evaluation only
        rgb, depth = perception.cam.render_rgbd(data)         # static scene: render once, fresh noise per frame
        target, first = TargetFilter(), None
        for _ in range(n_frames):
            m = perception.measure_images(rgb, depth, data)
            first = first or m
            x, P = target.update(m)
        trials.append(None if first is None else {
            "e1": first.p_world - p_true, "P1": first.cov_world,      # single frame
            "ef": x - p_true, "Pf": P,                                # filtered
            "offset": np.linalg.norm(first.uv - perception.K[:2, 2])})  # px from the image centre
    return trials


if __name__ == "__main__":
    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    perception = ObjectPerception(model, seed=0)
    cm.pose_arm_to_look_down(model, data, perception.cam.cam_id)

    # ---- 1. randomised trials at nominal noise
    trials = run_trials(model, data, perception, np.random.default_rng(0), N_TRIALS, N_FRAMES)
    ok = [t for t in trials if t is not None]
    E1 = 1000 * np.array([t["e1"] for t in ok])                        # mm
    EF = 1000 * np.array([t["ef"] for t in ok])
    nees1 = np.array([nees(t["e1"], t["P1"]) for t in ok])
    neesf = np.array([nees(t["ef"], t["Pf"]) for t in ok])

    print(f"detected in {len(ok)} of {N_TRIALS} trials")
    print(f"{'':18s}{'RMS x, y, z [mm]':>24s}{'|e| mean':>10s}{'|e| 95%':>9s}{'ANEES':>8s}{'NEES<7.8':>10s}")
    for name, E, N in ((f"single frame", E1, nees1), (f"filtered ({N_FRAMES})", EF, neesf)):
        n = np.linalg.norm(E, axis=1)
        print(f"{name:18s}{str(np.round(np.sqrt((E ** 2).mean(axis=0)), 2)):>24s}"
              f"{n.mean():10.2f}{np.percentile(n, 95):9.2f}{N.mean():8.2f}{(N < CHI2_3_95).mean():10.0%}")

    # ---- 2. noise sweep (single frames)
    sweep = []
    for k in NOISE_SCALES:
        set_noise_scale(k)
        seen = [t for t in run_trials(model, data, perception, np.random.default_rng(1), SWEEP_TRIALS, 1) if t]
        rms = 1000 * np.sqrt(np.mean([t["e1"] ** 2 for t in seen], axis=0)) if seen else np.full(3, np.nan)
        sweep.append((k, rms, len(seen)))
    set_noise_scale(1)
    perception.close()
    print("noise scale -> RMS x, y, z [mm] (detected):",
          "  ".join(f"{k}x {np.round(r, 2)} ({n}/{SWEEP_TRIALS})" for k, r, n in sweep))

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for i, axis in enumerate("xyz"):
        ax[0, 0].hist(E1[:, i], bins=20, alpha=0.5, label=f"{axis}")
    ax[0, 0].set(title=f"single-frame error ({len(ok)} trials)", xlabel="error [mm]", ylabel="count")
    ax[0, 0].legend()

    ks = [k for k, _, _ in sweep]
    for i, axis in enumerate("xyz"):
        ax[0, 1].plot(ks, [r[i] for _, r, _ in sweep], "o-", label=axis)
    ax[0, 1].set(title="noise sweep", xlabel="noise scale (x nominal)", ylabel="RMS error [mm]")
    ax[0, 1].legend()

    bins = np.linspace(0, max(15, nees1.max(), neesf.max()), 30)
    ax[1, 0].hist(nees1, bins=bins, alpha=0.5, label=f"single (ANEES {nees1.mean():.2f})")
    ax[1, 0].hist(neesf, bins=bins, alpha=0.5, label=f"filtered (ANEES {neesf.mean():.2f})")
    ax[1, 0].axvline(CHI2_3_95, color="k", ls="--", label="95% bound")
    ax[1, 0].axvline(3, color="g", ls=":", label="ideal mean 3")
    ax[1, 0].set(title="consistency: NEES", xlabel="NEES")
    ax[1, 0].legend()

    ax[1, 1].scatter([t["offset"] for t in ok], np.linalg.norm(E1, axis=1), s=12)
    ax[1, 1].set(title="error vs object position in the image", xlabel="distance from image centre [px]",
                 ylabel="|error| [mm]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    plt.tight_layout()
    cm.RESULTS.mkdir(exist_ok=True)
    plt.savefig(cm.RESULTS / "evaluation.png", dpi=120)
    plt.show()