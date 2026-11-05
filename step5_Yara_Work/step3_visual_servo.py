# This code uses the UR3e wrist camera to detect a cube and move the robot's end effector to a position 15 cm above it.
# Aalia's perception system estimates the cube position. Sameeha's Kalman filter estimates robot joint states.
# PBVS calculates joint velocities using the robot Jacobian. LQR tracks the desired joint positions and velocities.
#The robot stops approaching when its estimated Cartesian position error is below 1 cm.

import time
import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env

from step3_Aalia_Work.step6_perception import (ObjectPerception, TargetFilter)

from step3_Aalia_Work.step1_camera import (pose_arm_to_look_down)

from step4_Sameeha_Work.observer import (JointObserver, add_encoder_sensors)

from step4_Sameeha_Work.step1_state_space_model import (Arm, linearise)

from scipy.linalg import solve_discrete_are

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


# PBVS settings (Position-based visual servoing)
# PBVS_GAIN controls how strongly the robot responds to position error.
# DAMPING improves numerical behaviour near singularities.
# MAX_JOINT_SPEED limits commanded joint velocities [rad/s].

PBVS_GAIN = 0.8
DAMPING = 0.05
MAX_JOINT_SPEED = 0.35

# Approach target is 0.15 m above the cube.
APPROACH_HEIGHT = 0.15

# Target is reached when Cartesian error is below 0.01 m.
POSITION_TOLERANCE = 0.01

# Pereception settings (Camera and RGB-D sensing)
# Update object detection every 0.1 s (10 Hz).
# Require three successful measurements before approaching.

PERCEPTION_PERIOD = 0.1
MAX_TIME = 30.0
REQUIRED_DETECTIONS = 3
#these numbers were also picked through trail and error and comparing results.


# LQR settings 

Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0


# Joint Encoders (Proprioceptive sensing, Encoders provide measurements for the Kalman filter)

original_compile = mujoco.MjSpec.compile

def compile_with_encoders(spec):

    # Generate sensor names using the six joint names.
    encoder_names = [
        "enc_" + name
        for name in JOINT_NAMES
    ]

    # Read existing sensor names.
    existing_names = [
        sensor.name
        for sensor in spec.sensors
]

    # Add encoders if they are not already present.
    if not all(
        name in existing_names
        for name in encoder_names
    ):

        # Prevent adding duplicate or incomplete sensor sets.
        if any(
            name in existing_names
            for name in encoder_names
        ):
            raise RuntimeError(
                "Only some encoder sensors exist."
            )

        add_encoder_sensors(spec)

    # Compile model with encoder sensors.
    return original_compile(spec)


# ---- Load Sim
# Build the shared UR3e scene with encoders.

mujoco.MjSpec.compile = compile_with_encoders

try:
    model = env.build_scene()
finally:
    mujoco.MjSpec.compile = original_compile

data = mujoco.MjData(model)

# Reset the robot to its home configuration.
env.reset_home(model, data)

# Randomise the cube position using a fixed seed.
env.randomise_object(model, data, np.random.default_rng(1))

# Find joint & actuator indices:
# Identify positions, velocities, actuators and end-effector site.

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

# End-effector site used for Cartesian position control.
ee_site_id = model.site("attachment_site").id

# Read actuator command limits.
lower = model.actuator_ctrlrange[actuator_ids, 0]

upper = model.actuator_ctrlrange[actuator_ids, 1]


# Initialise camera & perception (Camera and visual features | RGB-D depth sensing)
# Detect the cube and estimate its 3D world position.

perception = ObjectPerception(model, seed=1)
target_filter = TargetFilter()

# Move the wrist camera into a downward-looking configuration.
pose_arm_to_look_down(model, data, perception.cam.cam_id)

# Save the initial viewing configuration.
view_q = data.qpos[q_ids].copy()

# Hold the robot at the viewing pose.
data.ctrl[actuator_ids] = view_q

# Update kinematics and sensor values.
mujoco.mj_forward(model, data)


# Kalman Filter:
# Estimate six joint positions and six joint velocities.

observer = JointObserver(model, q0=view_q, seed=0)

print("Kalman filter initialised")
print("Encoder sensor indices:", observer.sensor_adr)


# building the state-spcae mdle 
# Linearise the UR3e around the camera viewing pose.
# x = [q1...q6, dq1...dq6]

arm = Arm(model)

Ad, Bd, x_star, u_star = linearise(arm, view_q)

# design LQR controller 
# Minimise state error while limiting control corrections.

# Penalise joint position and velocity errors.
Q = np.diag([Q_POSITION] * 6 + [Q_VELOCITY] * 6)

# Penalise actuator control effort.
R = np.diag([R_CONTROL] * 6)

# Solve the discrete algebraic Riccati equation.
P_lqr = solve_discrete_are(Ad, Bd, Q, R)

# Calculate LQR feedback gain.
K_lqr = np.linalg.solve(R + Bd.T @ P_lqr @ Bd, Bd.T @ P_lqr @ Ad)

# check the LQR stability (Closed-loop eigenvalues)
# Discrete-time stability requires |eigenvalue| < 1.

closed_loop = Ad - Bd @ K_lqr

max_eigenvalue = np.max(np.abs(np.linalg.eigvals(closed_loop)))

print("LQR maximum eigenvalue:", max_eigenvalue)

if max_eigenvalue >= 1.0:
    raise RuntimeError("LQR linearised closed-loop system is unstable")

# ---- est ee state (Forward kinematics and Jacobian,  Use Kalman-estimated joint positions)
# Calculate end-effector position and translational Jacobian.

estimate_data = mujoco.MjData(model)

def get_estimated_ee_state(q_hat):
    # Copy current simulation configuration.
    estimate_data.qpos[:] = data.qpos

    # Replace robot angles with Kalman-estimated positions.
    estimate_data.qpos[q_ids] = q_hat

    # Set estimated-model velocities to zero.
    estimate_data.qvel[:] = 0.0

    # Recalculate forward kinematics.
    mujoco.mj_forward(model, estimate_data)

    # Estimated end-effector position [x, y, z].
    ee_position = estimate_data.site_xpos[ee_site_id].copy()

    # Allocate translational and rotational Jacobians.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Calculate end-effector Jacobian.
    mujoco.mj_jacSite(model, estimate_data, jacp, jacr, ee_site_id)

    # Extract the 3x6 translational Jacobian.
    J = jacp[:, v_ids]

    return ee_position, J

# PBVS controller (Position-based visual servoing):
# Cartesian error: e = target_position - ee_position
# Desired velocity: v = PBVS_GAIN * e
# Jacobian relation: v = J * qdot
# Damped least squares: qdot = J.T * (J*J.T + lambda^2*I)^-1 * v

def pbvs_controller(target_position, x_hat):

    # Extract estimated joint positions.
    q_hat = x_hat[:6]

    # Calculate estimated end-effector position and Jacobian.
    ee_position, J = get_estimated_ee_state(q_hat)

    # Calculate Cartesian position error [m].
    position_error = (target_position - ee_position)

    # Convert position error to desired Cartesian velocity [m/s].
    v_desired = (PBVS_GAIN * position_error)

    # Damped least-squares matrix.
    A = (J @ J.T + DAMPING**2 * np.eye(3))

    # Convert Cartesian velocity into joint velocities [rad/s].
    qdot_command = J.T @ np.linalg.solve(A, v_desired)

    # Limit commanded joint velocities.
    qdot_command = np.clip(qdot_command, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)

    return qdot_command, position_error

# ---- LQR controller
# Control: u = q_desired + K_lqr*(x_desired - x_hat)

def lqr_controller(q_desired, qdot_desired, x_hat):

    # Desired 12-state vector.
    x_desired = np.concatenate([q_desired, qdot_desired])

    # Compare desired and estimated states.
    state_error = (x_desired - x_hat)

    # Calculate corrected joint-position commands.
    u = (q_desired + K_lqr @ state_error)

    return u

# ----Intialise control variables
# Store target estimates, detection count and controller references.

print("\nUR3e PBVS + Kalman + LQR started")

next_perception = 0.0
next_print = 0.0

target = None
target_cov = None

detection_count = 0
reached_target = False

q_desired = view_q.copy()
qdot_desired = np.zeros(6)

clip_count = 0

# ---- main visual servoing loop
# Integrate perception, estimation and control.

try:

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while (
            viewer.is_running()
            and data.time < MAX_TIME
        ):

            step_start = time.time()

            # ---- camera perception update
            # Detect cube using RGB-D sensing.
            # Update every 0.1 seconds (10 Hz).

            if data.time >= next_perception:

                next_perception += PERCEPTION_PERIOD

                measurement = perception.measure(data)

                # Filter the measured cube position.
                target, target_cov = (target_filter.update(measurement))

                # Count successful detections.
                if measurement is not None:
                    detection_count += 1

            # ---- kalman filter update
            # Estimate joint positions and velocities.
            # x_hat contains 12 estimated robot states.

            x_hat, P = observer.update(data.sensordata, data.ctrl)

            # ---- reset desired velocities: Hold position until a valid target is available.

            qdot_desired[:] = 0.0

            # ----check cube detection: Require at least three successful measurements.

            if (
                target is not None
                and detection_count >= REQUIRED_DETECTIONS
            ):
                # Move to 0.15 m above the detected cube.
                target_position = target.copy()

                target_position[2] += (APPROACH_HEIGHT)

                # ---- calc PBVS command
                #Convert Cartesian error into joint velocities.

                qdot_command, position_error = (pbvs_controller(target_position,x_hat))

                # ---- check position error
                # Euclidean distance: ||e|| = sqrt(ex^2 + ey^2 + ez^2)

                error_norm = np.linalg.norm(position_error)

                reached_target = (error_norm < POSITION_TOLERANCE)

                # ---- update desired joint postion:
                # Integrate joint velocities: q_next = q + qdot*dt

                if not reached_target:

                    qdot_desired = qdot_command.copy()

                    q_desired += (qdot_desired * model.opt.timestep)

                # Keep desired joint positions within actuator limits.
                q_desired = np.clip(q_desired, lower, upper)

                # ---- Print position errors
                # Display error and target once per second.

                if data.time >= next_print:

                    next_print += 1.0

                    print(
                        f"t={data.time:.1f}s | "
                        f"error={error_norm:.4f}m | "
                        f"target={np.round(target_position, 3)}"
                    )

            # ---- LQR state-feedback control
            # Track desired states using Kalman estimates.

            u = lqr_controller(q_desired, qdot_desired, x_hat)

            # ---- enforce actuator limits 
            # Clip commands to the allowed actuator range.

            u_before_clip = u.copy()

            u = np.clip(u, lower, upper)

            if np.any(
                np.abs(u_before_clip - u) > 1e-8
            ):
                clip_count += 1

            # Send corrected commands to the UR3e.
            data.ctrl[actuator_ids] = u


            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start

            remaining = (
                model.opt.timestep - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

finally:

    # Release camera rendering resources.
    perception.close()

# Report whether the target was reached and actuator clipping.

print("\nPBVS simulation finished")

print("Target reached:", reached_target)

print("Actuator clipping steps:", clip_count)

print(
    "Final estimated joint positions:",
    np.round(x_hat[:6], 4)
)