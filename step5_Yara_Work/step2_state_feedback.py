#This code controls the UR3e using a discrete LQR controller and Sameeha's Kalman filter. The robot follows a smooth home-to-target-and-back trajectory.
#The Kalman filter estimates joint positions and velocities from simulated encoder measurements, while LQR uses these estimates to calculate corrective joint commands.
#to calculate corrective joint commands. and control commands using graphs and RMS errors.

import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

import step1_Main.environment as env

from scipy.linalg import solve_discrete_are

from step4_Sameeha_Work.step1_state_space_model import (Arm, linearise)
from step4_Sameeha_Work.observer import (JointObserver, add_encoder_sensors)


# ---- Adding the encoder sensors (Proprioceptive sensing)
# Encoders measure the six robot joint positions. ---> These measurements are used by the Kalman filter.

original_compile = mujoco.MjSpec.compile

def compile_with_encoders(spec):

    encoder_names = ["enc_shoulder_pan_joint", "enc_shoulder_lift_joint", "enc_elbow_joint", "enc_wrist_1_joint", "enc_wrist_2_joint", "enc_wrist_3_joint"]

    # Check which sensors already exist in the model.
    existing_names = [sensor.name for sensor in spec.sensors]

    # Add encoders only if none of them already exist.
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

        # Sameeha's function adds six joint-position sensors.
        add_encoder_sensors(spec)

    # Compile the model using MuJoCo's original function.
    return original_compile(spec)

# Load the shared Enviro
# Temporarily add encoders while building the MuJoCo model.

mujoco.MjSpec.compile = compile_with_encoders

try:
    model = env.build_scene()
finally:
    # Restore the original compile function.
    mujoco.MjSpec.compile = original_compile

data = mujoco.MjData(model)

# Reset robot to its home configuration.
env.reset_home(model, data)


# Joints and Actuator definations
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

# Find joint position indices in MuJoCo's qpos array.
q_ids = np.array([
    model.joint(name).qposadr[0]
    for name in JOINT_NAMES
])

# Find joint velocity indices in MuJoCo's qvel array.
v_ids = np.array([
    model.joint(name).dofadr[0]
    for name in JOINT_NAMES
])

# Find the six actuator indices.
actuator_ids = np.array([
    model.actuator(name).id
    for name in ACTUATOR_NAMES
])

# ---- Define movments 
# Robot moves home -> target -> home.

home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

target_q = np.array([-1.0, -1.2, 1.2, -1.0, -1.2, 0.5])

MOVE_TIME = 4.0
HOLD_TIME = 2.0

# Total cycle time = 2 * MOVE_TIME + 2 * HOLD_TIME
CYCLE_TIME = 2 * MOVE_TIME + 2 * HOLD_TIME

TOTAL_TIME = CYCLE_TIME


# LQR (Linear Quadratic Regulator) variables and setting
# Q penalises state errors; R penalises control effort.
# Larger Q_POSITION prioritises accurate joint positioning.

Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0
# these values were chosen by testing other values to determin the best 

# ---- State-space modelling
# Sameeha's function linearises the UR3e around home_q.
#
# x[k+1] = Ad*x[k] + Bd*u[k]  (deviation-state form)
# x = [q1...q6, dq1...dq6]  -> 12 states

arm = Arm(model)

Ad, Bd, x_star, u_star = linearise(arm, home_q)


# desiging the LQR controller 
# Minimise J = sum(x_error.T*Q*x_error + u_error.T*R*u_error)
# K_lqr determines how strongly the controller corrects errors.

# Penalise joint position and velocity errors.
Q = np.diag(
    [Q_POSITION] * 6 +
    [Q_VELOCITY] * 6
)

# Penalise excessive control commands.
R = np.diag([R_CONTROL] * 6)

# Solve the discrete algebraic Riccati equation.
P_lqr = solve_discrete_are(Ad, Bd, Q, R)

# Calculate the optimal LQR feedback gain matrix.
K_lqr = np.linalg.solve(R + Bd.T @ P_lqr @ Bd, Bd.T @ P_lqr @ Ad)

# Cheking the closed loop Stability (Closed-loop eigenvalues) 
# Discrete-time stability requires |eigenvalue| < 1.

closed_loop = Ad - Bd @ K_lqr

eigenvalues = np.linalg.eigvals(closed_loop)

print("LQR gain matrix:")
print(K_lqr)

print("Maximum closed-loop eigenvalue magnitude:")
print(np.max(np.abs(eigenvalues)))

if np.max(np.abs(eigenvalues)) >= 1.0:
    raise RuntimeError("LQR closed-loop system is not stable")


# Intialise Kalman Filter (State estimation and output feedbck)
# Estimate joint positions and velocities using encoder measurements.

observer = JointObserver(model, q0=home_q, seed=0)

print("Kalman filter initialised")
print("Encoder sensor indices:", observer.sensor_adr)

# ----Smooth Trajectory
# Cubic smoothstep: s = 3r^2 - 2r^3
# Returns desired joint positions & velocities.

def smooth_trajectory(start_q, end_q, t, duration):

    # Normalised movement progress (0 to 1).
    r = np.clip(t / duration, 0.0, 1.0)

    # Smooth interpolation factor.
    s = 3 * r**2 - 2 * r**3

    # Derivative of smoothstep with respect to time.
    s_dot = (6 * r - 6 * r**2) / duration

    # Desired joint positions [rad].
    q_desired = start_q + s * (end_q - start_q)

    # Desired joint velocities [rad/s].
    qdot_desired = s_dot * (end_q - start_q)

    return q_desired, qdot_desired

# ----LQR state feedback controller (State feedback, Estimated-state output feedback)
# State error = desired state - estimated state
# Control: u = q_desired + K_lqr * state_error

def state_feedback_controller(q_desired, qdot_desired, x_hat):

    # Combine desired positions and velocities into 12 states.
    x_desired = np.concatenate([q_desired, qdot_desired])

    # Calculate tracking error using Kalman estimates.
    state_error = x_desired - x_hat

    # Calculate corrective actuator position commands.
    u = q_desired + K_lqr @ state_error

    # Separate position and velocity errors.
    position_error = state_error[:6]
    velocity_error = state_error[6:]

    return u, position_error, velocity_error

# ---- Store the perf data
# Record desired, actual and estimated states for evaluation.

time_history = []

desired_history = []
actual_history = []
estimated_history = []

error_history = []
control_history = []

observer_error_history = []

# Start the sim (LQR and Kalman-filter-based control)

print("\nUR3e state-feedback test started")

print("Controller: Discrete LQR")
print("Q diagonal =", np.diag(Q))
print("R diagonal =", np.diag(R))

print("State estimation: Sameeha's Kalman filter")


# ---- Mian control loop
# Repeat trajectory generation, state estimation and feedback.

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < TOTAL_TIME:

        step_start = time.time()

        t = data.time

        # ---- Move home to target (0-4 s)
        if t < MOVE_TIME:
            q_desired, qdot_desired = smooth_trajectory(home_q, target_q, t, MOVE_TIME)

        # ---- hold target (4-6 s)
        elif t < MOVE_TIME + HOLD_TIME:
            q_desired = target_q.copy()
            qdot_desired = np.zeros(6)

        # ---- Return home (6-10 s)
        elif t < 2 * MOVE_TIME + HOLD_TIME:

            return_time = t - MOVE_TIME - HOLD_TIME

            q_desired, qdot_desired = smooth_trajectory(target_q, home_q, return_time, MOVE_TIME)

        # ---- Hole Home (10-12 s)
        else:
            q_desired = home_q.copy()
            qdot_desired = np.zeros(6)

        # ---- Kalman Filter Update
        # Use sensor measurements and actuator commands.
        # x_hat = estimated 12-state vector
        # P = state-estimation error covariance
        x_hat, P = observer.update(data.sensordata, data.ctrl)

        # Estimated joint positions [rad].
        q_hat = x_hat[:6]

        # Estimated joint velocities [rad/s].
        qdot_hat = x_hat[6:]

        # ---- LQR State-feddback control
        # Compare desired states against Kalman-estimated states.

        u, position_error, velocity_error = (state_feedback_controller(q_desired, qdot_desired, x_hat))

        # ---- Apply actuator limits
        # Prevent commands exceeding the actuator control range.

        lower = model.actuator_ctrlrange[actuator_ids, 0]

        upper = model.actuator_ctrlrange[actuator_ids, 1]

        u_before_clip = u.copy()

        u = np.clip(u, lower, upper)

        if np.any(np.abs(u_before_clip - u) > 1e-8):
            print("Warning: Actuator command clipped") # for debugging and checking testing results 

        # Send LQR commands to the six position actuators.
        data.ctrl[actuator_ids] = u

        # ---- read robot state (actual)
        # Actual MuJoCo states are used for evaluation only.

        q_actual = data.qpos[q_ids].copy()

        qdot_actual = data.qvel[v_ids].copy()

        actual_state = np.concatenate([
            q_actual,
            qdot_actual
        ])

        # ---- calc est and tracking errors
        # Kalman error = estimated state - actual state
        # Tracking error = desired position - actual position

        observer_error = x_hat - actual_state

        actual_tracking_error = (q_desired - q_actual)

        # ---- recored results
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

        # ---- update robot dynamics
        # Advance MuJoCo physics by one timestep.
        mujoco.mj_step(model, data)

        viewer.sync()

        elapsed = time.time() - step_start

        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)


# Conver the results to arrays:
# Convert recorded values into arrays for plotting and RMS calculations.

time_history = np.array(time_history)

desired_history = np.array(desired_history)
actual_history = np.array(actual_history)
estimated_history = np.array(estimated_history)

error_history = np.array(error_history)
control_history = np.array(control_history)

observer_error_history = np.array(
    observer_error_history
)


# Ploting the data on graphes.
# Evaluate tracking, control effort and estimation accuracy.

if len(time_history) > 0:
    # ---- GRAPH 1: Desired vs Actual vs Estimated
    # Compare commanded, true and Kalman-estimated joint positions.

    plt.figure(figsize=(10, 5))

    plt.plot(time_history, desired_history[:, 0], label="Desired shoulder pan")

    plt.plot(time_history, actual_history[:, 0], "--", label="Actual shoulder pan")

    plt.plot(time_history, estimated_history[:, 0], ":", label="Estimated shoulder pan (Kalman)")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title("Joint Position Tracking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---- GRAPH 2: Position tracking error:
    # Show desired-minus-actual error for all six joints.

    plt.figure(figsize=(10, 5))

    for i in range(6):

        plt.plot(time_history, error_history[:, i], label=JOINT_NAMES[i])

    plt.xlabel("Time (s)")
    plt.ylabel("Position error (rad)")
    plt.title("Joint Position Tracking Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---- GRAPH 3: contorl commands:
    # Show the commanded actuator positions for all six joints.

    plt.figure(figsize=(10, 5))

    for i in range(6):

        plt.plot(time_history, control_history[:, i], label=ACTUATOR_NAMES[i])

    plt.xlabel("Time (s)")
    plt.ylabel("Commanded joint position (rad)")
    plt.title("Position Actuator Control Commands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---- Calc RMS tracking error
    # RMSE = sqrt(mean(error^2))
    # Smaller RMSE means better joint-position tracking.

    rms_error = np.sqrt(
        np.mean(error_history**2, axis=0)
    )

    print("\n---- TRACKING RESULTS")

    for i in range(6):

        print(
            f"{JOINT_NAMES[i]}: "
            f"RMS error = {rms_error[i]:.5f} rad"
        )

    # ---- Calc Kalman filter accuracy
    # Compare estimated positions and velocities with actual states.

    observer_position_rms = np.sqrt(np.mean(observer_error_history[:, :6]**2, axis=0))

    observer_velocity_rms = np.sqrt(np.mean(observer_error_history[:, 6:]**2, axis=0))

    print("\n---- KALMAN FILTER RESULTS")

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