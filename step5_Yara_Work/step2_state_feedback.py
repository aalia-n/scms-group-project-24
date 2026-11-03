
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

import step1_Main.environment as env

from scipy.linalg import solve_discrete_are

from step4_Sameeha_Work.step1_state_space_model import (
    Arm,
    linearise
)

from step4_Sameeha_Work.observer import (
    JointObserver,
    add_encoder_sensors
)

original_compile = mujoco.MjSpec.compile


def compile_with_encoders(spec):

    encoder_names = ["enc_shoulder_pan_joint", "enc_shoulder_lift_joint", "enc_elbow_joint", "enc_wrist_1_joint", "enc_wrist_2_joint", "enc_wrist_3_joint"]

    existing_names = [
        sensor.name for sensor in spec.sensors
    ]

    if not all(
        name in existing_names
        for name in encoder_names
    ):
        if any(
            name in existing_names
            for name in encoder_names
        ):
            raise RuntimeError(
                "Only some encoder sensors exist. "
                "Check environment.py."
            )

        add_encoder_sensors(spec)

    return original_compile(spec)


mujoco.MjSpec.compile = compile_with_encoders

try:
    model = env.build_scene()
finally:
    mujoco.MjSpec.compile = original_compile


data = mujoco.MjData(model)

env.reset_home(model, data)

# JOINT AND ACTUATOR NAMES

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


q_ids = np.array([
    model.joint(name).qposadr[0]
    for name in JOINT_NAMES
])

v_ids = np.array([
    model.joint(name).dofadr[0]
    for name in JOINT_NAMES
])

actuator_ids = np.array([
    model.actuator(name).id
    for name in ACTUATOR_NAMES
])

home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

target_q = np.array([-1.0, -1.2, 1.2, -1.0, -1.2, 0.5])

MOVE_TIME = 4.0
HOLD_TIME = 2.0

CYCLE_TIME = 2 * MOVE_TIME + 2 * HOLD_TIME

TOTAL_TIME = CYCLE_TIME

# LQR TUNING SETTINGS

Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0

# CONTROLLER GAINS

# LQR CONTROLLER DESIGN

arm = Arm(model)

# Sameeha's linearised state-space model
Ad, Bd, x_star, u_star = linearise(arm, home_q)

# Initial LQR weighting matrices
Q = np.diag(
    [Q_POSITION] * 6 +
    [Q_VELOCITY] * 6
)

R = np.diag([R_CONTROL] * 6)

# Solve the discrete algebraic Riccati equation
P_lqr = solve_discrete_are(Ad, Bd, Q, R)

# Calculate the optimal feedback gain
K_lqr = np.linalg.solve(
    R + Bd.T @ P_lqr @ Bd,
    Bd.T @ P_lqr @ Ad
)

# Check closed-loop stability
closed_loop = Ad - Bd @ K_lqr

eigenvalues = np.linalg.eigvals(closed_loop)

print("LQR gain matrix:")
print(K_lqr)

print("Maximum closed-loop eigenvalue magnitude:")
print(np.max(np.abs(eigenvalues)))

if np.max(np.abs(eigenvalues)) >= 1.0:
    raise RuntimeError("LQR closed-loop system is not stable")

# INITIALISE SAMEEHA'S KALMAN FILTER
observer = JointObserver(
    model,
    q0=home_q,
    seed=0
)

print("Kalman filter initialised")
print("Encoder sensor indices:", observer.sensor_adr)

# SMOOTH TRAJECTORY

def smooth_trajectory(start_q, end_q, t, duration):
    """
    Generate desired joint positions and velocities.

    Cubic trajectory:
        s = 3r^2 - 2r^3

    Its derivative gives the desired velocity.
    """

    r = np.clip(t / duration, 0.0, 1.0)

    s = 3 * r**2 - 2 * r**3

    s_dot = (6 * r - 6 * r**2) / duration

    q_desired = start_q + s * (end_q - start_q)

    qdot_desired = s_dot * (end_q - start_q)

    return q_desired, qdot_desired

# STATE-FEEDBACK CONTROLLER

# LQR STATE-FEEDBACK CONTROLLER

def state_feedback_controller(q_desired, qdot_desired, x_hat):

    # Desired 12-state vector
    x_desired = np.concatenate([
        q_desired,
        qdot_desired
    ])

    # State tracking error
    state_error = x_desired - x_hat

    # LQR control with equilibrium feedforward
    # LQR trajectory tracking with position reference
    u = q_desired + K_lqr @ state_error

    position_error = state_error[:6]
    velocity_error = state_error[6:]

    return u, position_error, velocity_error

# STORAGE FOR RESULTS

time_history = []

desired_history = []
actual_history = []
estimated_history = []

error_history = []
control_history = []

observer_error_history = []

# START SIMULATION

print("\nUR3e state-feedback test started")

print("Controller: Discrete LQR")
print("Q diagonal =", np.diag(Q))
print("R diagonal =", np.diag(R))

print("State estimation: Sameeha's Kalman filter")


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < TOTAL_TIME:

        step_start = time.time()

        t = data.time

        # GENERATE DESIRED TRAJECTORY

        if t < MOVE_TIME:

            q_desired, qdot_desired = smooth_trajectory(
                home_q,
                target_q,
                t,
                MOVE_TIME
            )

        elif t < MOVE_TIME + HOLD_TIME:

            q_desired = target_q.copy()
            qdot_desired = np.zeros(6)

        elif t < 2 * MOVE_TIME + HOLD_TIME:

            return_time = t - MOVE_TIME - HOLD_TIME

            q_desired, qdot_desired = smooth_trajectory(
                target_q,
                home_q,
                return_time,
                MOVE_TIME
            )

        else:

            q_desired = home_q.copy()
            qdot_desired = np.zeros(6)

        # SAMEEHA: KALMAN FILTER
        x_hat, P = observer.update(
            data.sensordata,
            data.ctrl
        )

        # Estimated joint positions
        q_hat = x_hat[:6]

        # Estimated joint velocities
        qdot_hat = x_hat[6:]


        # YARA: STATE-FEEDBACK CONTROL

        u, position_error, velocity_error = (
            state_feedback_controller(
                q_desired,
                qdot_desired,
                x_hat
            )
        )

        # ACTUATOR LIMITS

        lower = model.actuator_ctrlrange[
            actuator_ids, 0
        ]

        upper = model.actuator_ctrlrange[
            actuator_ids, 1
        ]

        u_before_clip = u.copy()

        u = np.clip(u, lower, upper)

        if np.any(np.abs(u_before_clip - u) > 1e-8):
            print("Warning: Actuator command clipped")

        data.ctrl[actuator_ids] = u


        # RECORD ACTUAL STATES FOR EVALUATION

        q_actual = data.qpos[q_ids].copy()

        qdot_actual = data.qvel[v_ids].copy()

        actual_state = np.concatenate([
            q_actual,
            qdot_actual
        ])

        observer_error = x_hat - actual_state

        actual_tracking_error = (
            q_desired - q_actual
        )

        time_history.append(data.time)

        desired_history.append(q_desired.copy())

        actual_history.append(q_actual.copy())

        estimated_history.append(q_hat.copy())

        error_history.append(
            actual_tracking_error.copy()
        )

        control_history.append(u.copy())

        observer_error_history.append(
            observer_error.copy()
        )

        mujoco.mj_step(model, data)

        viewer.sync()

        elapsed = time.time() - step_start

        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)

# CONVERT RESULTS TO ARRAYS

time_history = np.array(time_history)

desired_history = np.array(desired_history)
actual_history = np.array(actual_history)
estimated_history = np.array(estimated_history)

error_history = np.array(error_history)
control_history = np.array(control_history)

observer_error_history = np.array(
    observer_error_history
)

# PLOT RESULTS

if len(time_history) > 0:

    # GRAPH 1: DESIRED VS ACTUAL POSITION

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_history,
        desired_history[:, 0],
        label="Desired shoulder pan"
    )

    plt.plot(
        time_history,
        actual_history[:, 0],
        "--",
        label="Actual shoulder pan"
    )

    plt.plot(
        time_history,
        estimated_history[:, 0],
        ":",
        label="Estimated shoulder pan (Kalman)"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title("Joint Position Tracking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # GRAPH 2: POSITION TRACKING ERROR

    plt.figure(figsize=(10, 5))

    for i in range(6):

        plt.plot(
            time_history,
            error_history[:, i],
            label=JOINT_NAMES[i]
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Position error (rad)")
    plt.title("Joint Position Tracking Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # GRAPH 3: CONTROL COMMANDS

    plt.figure(figsize=(10, 5))

    for i in range(6):

        plt.plot(
            time_history,
            control_history[:, i],
            label=ACTUATOR_NAMES[i]
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Commanded joint position (rad)")
    plt.title("Position Actuator Control Commands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # RMS TRACKING ERROR

    rms_error = np.sqrt(
        np.mean(error_history**2, axis=0)
    )

    print("\n========== TRACKING RESULTS ==========")

    for i in range(6):

        print(
            f"{JOINT_NAMES[i]}: "
            f"RMS error = {rms_error[i]:.5f} rad"
        )

    # KALMAN FILTER ACCURACY

    observer_position_rms = np.sqrt(
        np.mean(
            observer_error_history[:, :6]**2,
            axis=0
        )
    )

    observer_velocity_rms = np.sqrt(
        np.mean(
            observer_error_history[:, 6:]**2,
            axis=0
        )
    )

    print("\n========== KALMAN FILTER RESULTS ==========")

    for i in range(6):

        print(
            f"{JOINT_NAMES[i]}: "
            f"position RMS = "
            f"{observer_position_rms[i]:.5f} rad | "
            f"velocity RMS = "
            f"{observer_velocity_rms[i]:.5f} rad/s"
        )

    print("\nSimulation completed!")

    plt.show()

