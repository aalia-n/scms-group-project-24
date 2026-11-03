# Task 2 - Modelling and state estimation (Sameeha)

This folder is my part of the project. I made a state-space model of the UR3e, checked that it's controllable and observable, and built an observer (Kalman filter) that uses the 6 joint encoders to estimate all 12 states, so the 6 joint angles and the 6 joint speeds, $\hat x = [\hat q;\ \hat{\dot q}]$. The encoders only measure angles, so the speeds have to be estimated. The idea is that Yara's controller uses my estimate instead of MuJoCo's true values.

## My files

I did it in steps like the tutorials. Run them in order from the repo root (with the `.venv` on), or just press ▶ on each one in VS Code. Each one prints results and saves its graphs in `results/`.

| File | What it does | Criterion |
|---|---|---|
| `step1_state_space_model.py` | sets up x, u and y, shows M(q) changes with pose, makes the textbook model and the MuJoCo model (the one I use), checks it against MuJoCo | A |
| `step2_controllability_observability.py` | rank tests, Gramians, PBH, and what happens if a sensor breaks | A |
| `step3_measurement_and_ground_truth.py` | encoder noise, MuJoCo sensor timing, and the test motion I compare everything to | B / C |
| `step4_observers.py` | finite difference vs Luenberger vs Kalman filter | C |
| `step5_evaluation.py` | compares estimates to the true values (graphs, ±2σ, error table, NEES) | C |
| `step6_tuning.py` | picks the observer settings and tests different encoder noise | C |
| `step7_robustness.py` | push, unknown payload, 20 random trials, re-linearising | E |
| `observer.py` | my finished Kalman filter that `main.py` will import. Running it tests it in a copy of the main.py loop | - |
| `step8_testing.py` | 24 automatic PASS/FAIL checks + a summary graph and report | all |
| `run_all.py` | runs everything above in order (takes about a minute) | - |
| `step9_live_demo.py` | opens the MuJoCo viewer so you can watch my observer working live (push + payload). Prints the errors every second and saves `results/step9_live_demo.png` when you close it. Not in `run_all.py` because it needs the window | E |

## How to test it

From the repo root:

```
python step4_Sameeha_Work\run_all.py
```

This runs steps 1-7, `observer.py` and `step8_testing.py`, saves all the graphs to `results/`, and should finish with `RESULT: 24 / 24 checks passed`. If you only want the checks, run `python step4_Sameeha_Work\step8_testing.py`.

I set the pass marks based on what the task needs, not on what my code got. The main one is that the angle error has to be below the raw encoder noise, because if the observer can't beat just reading the encoder then there's no point using it. Every run saves the full table to `results/step8_test_report.md`.

What the checks cover:

- **Modelling (A)** - my linear model matches MuJoCo (>= 95 %), the arm holds still at home, it's stable, controllable and observable (both models), detectable and stabilisable
- **Sensing (B)** - the encoder noise matches the $R_n$ the filter uses, and the same seed always gives the same data
- **Estimation (C)** - the KF beats the raw encoder, is at least 50x better than finite difference on speed and better than Luenberger, every joint is under 0.01 rad/s, and >= 95 % of the errors are inside its own ±2σ
- **Robustness (E)** - other pose, 1.5 kg payload, steady load, push (and recovering after), 20 random trials, and the KF beats "model only" whenever there's a load
- **Integration** - `observer.py` gives exactly the same answers as step 4, works with the `update(sensordata, ctrl)` line from `main.py`, takes < 2 ms per update, and gives a proper error message if the sensors are missing

For the main.py check I add the encoder sensors inside the test only, so I haven't changed `environment.py` or `main.py`.

## Results

All **24 / 24 checks passed**. Encoder noise is 1.006 mrad and dt = 2 ms. The observers always use the normal model (linearised at home), even when the real robot is different, so they don't know about the push or the payload.

![Task 2 summary](results/step8_summary.png)

Joint angles (RMSE in mrad, raw encoder noise = 1.006):

| Test | model only | Luenberger | Kalman filter |
|---|---|---|---|
| normal | 0.218 | 0.213 | **0.165** |
| other pose | 0.360 | 0.217 | **0.170** |
| 0.5 kg payload | 0.911 | 0.248 | **0.205** |
| 1.0 kg payload | 1.959 | 0.332 | **0.294** |
| 1.5 kg payload | 3.007 | 0.429 | **0.394** |
| steady load | 2.188 | 0.367 | **0.393** |
| push | 0.405 | 0.227 | **0.209** |

Joint speeds (these are never measured, RMSE in rad/s):

| Test | finite difference | Luenberger | Kalman filter |
|---|---|---|---|
| normal | 0.709 | 0.0031 | **0.0016** |
| 1.0 kg payload | 0.712 | 0.0108 | **0.0085** |
| 1.5 kg payload | 0.713 | 0.0156 | **0.0128** |
| steady load | 0.712 | 0.0127 | **0.0097** |
| push | 0.719 | 0.0043 | **0.0027** |

20 random trials (start pose moved up to ±0.5 rad, payload 0-1 kg): KF angle error median 0.214 mrad, worst 0.275 mrad.

What I got from this:

1. The Kalman filter's angle estimate beat the raw encoder in every test, by about 2.5-6x.
2. The speeds came out 50-440x more accurate than finite difference (0.0016-0.013 rad/s vs about 0.71 rad/s), even though they're never measured.
3. The encoders are what make it robust. "Model only" (just running the model with no encoder correction) is fine when the model is right, but as soon as there's a load its angle error goes up to 3 mrad and stays there. The KF stays under 0.4 mrad.
4. It's not perfect. The KF assumes small random disturbances, so with a steady load Luenberger is slightly better on angle (0.367 vs 0.393 mrad), and the KF reacts slower to a sudden push (step 7). I also tried re-linearising the model as the arm moves but it didn't help, because the servos are stiff enough that one model is fine (step 7, E.4).

The graphs from every step (`step1_*.png` ... `step7_*.png`) are in `results/` too.

## Packages

```
pip install -r step1_Main/requirements.txt     # updates .venv to the team's versions
pip install scipy control                      # only needed for steps 4-7
```

`observer.py` only needs numpy and mujoco, so `main.py` won't need anything new.

## Integration (not done yet)

This needs a change to the shared `environment.py`, so I'm leaving it until the group is ready.

`main.py` calls `observer.update(data.sensordata, data.ctrl)`, but right now the scene doesn't have any sensors, so `data.sensordata` is empty. This needs to go in `build_scene()` in `step1_Main/environment.py`, just before `return spec.compile()`:

```python
    # encoders for the observer (Task 2, step4_Sameeha_Work): one jointpos sensor per arm joint
    for j in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]:
        spec.add_sensor(name="enc_" + j, type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=j)
```

(`environment.py` can't just import `observer.py` for this, because `observer.py` already imports `environment.py` and it'd be a circular import.)

After that, the three observer lines in `main.py` just need uncommenting:

```python
from step4_Sameeha_Work.observer import JointObserver
observer = JointObserver(model)
x_hat, P = observer.update(data.sensordata, data.ctrl)      # x_hat[:6] = angles, x_hat[6:] = speeds
```

`observer.update` needs to stay before the controller sets `data.ctrl` (it already is in `main.py`). After `mj_step`, `sensordata` is one step old and `data.ctrl` is the command from that step, and the observer handles both so the estimate is for right now.

## My settings

`W_O = 49 rad/s` for Luenberger and `Q_A = 100` for the Kalman filter, both picked in step 6. The encoder noise is $\sigma = 10^{-3}$ rad plus the rounding from a 14-bit encoder. I made it worse than a real UR encoder on purpose so the methods actually look different. If any of these change, they need changing in both `step4_observers.py` and `observer.py`.

## Tutorials I used

| Step | Based on | What I added for our robot |
|---|---|---|
| 1 | Week 5 (state space, linearisation), Week 6 (exact ZOH), Week 7 C.2 (Jacobians) | linearising the actual team scene with `mjd_transitionFD`, the hold command $u^\star$ from `mj_inverse`, explaining the servo poles, checking the model against MuJoCo |
| 2 | Week 5 (ctrb/obsv), Week 8 A (Gramians, PBH, stabilisable/detectable) | discrete-time Gramians and PBH, "what if an encoder fails" |
| 3 | Week 2 A (encoders as quantisers, $\Delta^2/12$) | the MuJoCo one-step sensor delay, a test setup with the same timing as `main.py` |
| 4 | Week 7 A (`ct.place`, duality), Week 7 B (`kf_step`, $Q_d$/$R_n$, `ct.lqe`) | prediction-form observers, $Q_d$ from random accelerations, leaving the fast servo poles alone |
| 5 | Week 7 (truth vs estimate, ±2σ, NEES) | ±2σ check per joint, χ² band for NEES, a CSV for the project page |
| 6 | Week 6 (trade-off curves), Week 7 (slow vs fast observer) | tuning on a mix of situations, one score for accuracy, accuracy vs honesty |
| 7 | Week 8 B (testing under noise and disturbance, limits of separation) | unknown payload, random poses, re-linearising the model as it moves |
