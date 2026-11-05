# This code integrates Aalia's RGB-D perception, Sameeha's Kalman state estimator, and Yara's PBVS and LQR controllers.
# The UR3e detects a cube, approaches it, performs a simulated grasp, lifts it, transports it to the drop zone, releases it,
# retracts its arm and returns to its starting viewing position.
# A finite-state machine coordinates each stage. The code also records performance data for Step 5.

import contextlib
import time

import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are

import step1_Main.environment as env
from step3_Aalia_Work.step6_perception import ObjectPerception, TargetFilter
from step3_Aalia_Work.step1_camera import pose_arm_to_look_down
from step4_Sameeha_Work.observer import JointObserver, add_encoder_sensors
from step4_Sameeha_Work.step1_state_space_model import Arm, linearise

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

# ---- Position-based visual servoing
# Gain controls the response to Cartesian position error.
# Damping helps near singularities; speed limits prevent large commands.
PBVS_GAIN = 0.8
DAMPING = 0.05
MAX_JOINT_SPEED = 0.35
POSITION_TOLERANCE = 0.012   # Normal movement tolerance: 12 mm.
PLACE_TOLERANCE = 0.002      # Placement tolerance: 2 mm.
PLACE_CLEARANCE = 0.0005     # Desired clearance between cube bottom and table: 0.5 mm.

# ---- RGB-D camera and object detection
# Camera measurements are processed at 10 Hz.
# Three consecutive successful detections are required.
PERCEPTION_PERIOD = 0.1
REQUIRED_DETECTIONS = 3

# ---- Simulation settings
MAX_TIME = 30.0
SETTLE_TIME = 5.0
STABLE_STEPS = 20

# ---- Pick-and-place settings (Cartesian target positions)
# Heights are measured relative to the detected cube centre.
APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.035
LIFT_HEIGHT = 0.18
DROP_XY = np.array([0.25, -0.20])   # Destination coordinates [x, y] in metres.

# ---- Home position settings
# Return to the original camera viewing configuration.
RETRACT_HEIGHT = 0.15
HOME_TOLERANCE = 0.02
HOME_STABLE_STEPS = 20
HOME_MAX_SPEED = 0.35

# ---- LQR weights
Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0

# Record controller and estimation results every 0.05 seconds.
LOG_PERIOD = 0.05

# ---- Pick-and-place state machine (integration and sequential task execution)
# Each stage changes only after its target is reached.
NEXT_STAGE = {"APPROACH": "LOWER", "LOWER": "LIFT", "LIFT": "TRANSPORT",
              "TRANSPORT": "PLACE", "PLACE": "RETRACT", "RETRACT": "RETURN_HOME"}

# Height above the cube used during each cube-related stage.
HEIGHT_ABOVE_CUBE = {"APPROACH": APPROACH_HEIGHT, "LOWER": GRASP_HEIGHT, "LIFT": LIFT_HEIGHT}


# ---- Encoder model (proprioceptive sensing | encoders provide measurements for the Kalman filter)
def build_model():
    original_compile = mujoco.MjSpec.compile

    # Generate encoder names from the robot joint names.
    encoder_names = ["enc_" + name for name in JOINT_NAMES]

    def compile_with_encoders(spec):
        # Check which sensors already exist.
        existing = {sensor.name for sensor in spec.sensors}
        present = [name in existing for name in encoder_names]

        # Add encoders only when the full set is missing.
        if not all(present):
            if any(present):
                raise RuntimeError("Only some encoder sensors exist.")
            add_encoder_sensors(spec)

        return original_compile(spec)

    # Temporarily extend model compilation with encoder sensors.
    mujoco.MjSpec.compile = compile_with_encoders
    try:
        return env.build_scene()
    finally:
        mujoco.MjSpec.compile = original_compile


# ---- Main simulation function
# Runs the full integrated pick-and-place task.
# log_performance: enable recording for Step 5.
# cube_seed: select a reproducible random cube position.
# show_viewer: display the MuJoCo simulation.
def run_simulation(log_performance=False, cube_seed=1, show_viewer=True):

    print("\nSTEP 4 - PICK AND PLACE")

    model = build_model()
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    env.reset_home(model, data)

    # Place the cube at a reproducible random position.
    env.randomise_object(model, data, np.random.default_rng(cube_seed))

    # ---- Identify joints and end effector
    # Locate robot joint positions, velocities and actuators.
    q_ids = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES])
    v_ids = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES])
    actuator_ids = np.array([model.actuator(n).id for n in ACTUATOR_NAMES])
    ee_site_id = model.site("attachment_site").id

    # ---- Identify cube
    # Free joint stores cube position, orientation and velocity.
    cube_joint = model.joint("target_free")
    cube_qadr = cube_joint.qposadr[0]
    cube_vadr = cube_joint.dofadr[0]
    cube_geom_id = model.geom("target_geom").id

    # ---- Actuator limits
    lower = model.actuator_ctrlrange[actuator_ids, 0]
    upper = model.actuator_ctrlrange[actuator_ids, 1]

    # ---- Drop-off position
    # Cube centre height = table height + cube half-height + clearance.
    drop_cube_position = np.array([DROP_XY[0], DROP_XY[1], env.TABLE_TOP_Z + env.OBJ_HALF + PLACE_CLEARANCE])

    # ---- Camera and perception
    # Aalia: RGB-D sensing and object localisation.
    # Initialise perception and move camera to a downward viewing pose.
    perception = ObjectPerception(model, seed=cube_seed)
    target_filter = TargetFilter()
    pose_arm_to_look_down(model, data, perception.cam.cam_id)

    # Save the robot's initial viewing joint positions.
    view_q = data.qpos[q_ids].copy()
    data.ctrl[actuator_ids] = view_q
    mujoco.mj_forward(model, data)

    # Return to this viewing pose after placing the cube.
    home_q = view_q.copy()

    # ---- Kalman observer
    # Sameeha: State estimation.
    # Estimate six joint positions and six joint velocities.
    observer = JointObserver(model, q0=view_q, seed=0)
    print("Kalman filter initialised")

    # ---- State-space model
    # Sameeha: State-space modelling.
    # Linearise the robot around the viewing configuration.
    # Delta_x[k+1] = Ad*Delta_x[k] + Bd*Delta_u[k]
    Ad, Bd, _, _ = linearise(Arm(model), view_q)

    # ---- LQR design
    # Yara: State-feedback control.
    # Minimise joint tracking errors and control corrections.
    Q = np.diag([Q_POSITION] * 6 + [Q_VELOCITY] * 6)
    R = np.diag([R_CONTROL] * 6)

    # Solve the discrete algebraic Riccati equation, then calculate the LQR feedback gain.
    P_lqr = solve_discrete_are(Ad, Bd, Q, R)
    K_lqr = np.linalg.solve(R + Bd.T @ P_lqr @ Bd, Bd.T @ P_lqr @ Ad)

    # ---- Linearised closed-loop stability
    # Discrete-time stability requires |eigenvalue| < 1.
    max_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K_lqr))))
    print("LQR maximum eigenvalue:", max_eigenvalue)
    if max_eigenvalue >= 1.0:
        raise RuntimeError("Linearised LQR system is unstable")

    # ---- Estimated end-effector state
    # Kinematics, estimation and Jacobian.
    estimate_data = mujoco.MjData(model)

    def get_estimated_ee_state(q_hat):
        # Use Kalman-estimated joint positions and update forward kinematics.
        estimate_data.qpos[:] = data.qpos
        estimate_data.qpos[q_ids] = q_hat
        estimate_data.qvel[:] = 0.0
        mujoco.mj_forward(model, estimate_data)

        # Calculate translational Jacobian.
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, estimate_data, jacp, None, ee_site_id)

        # Return end-effector position, rotation matrix and 3x6 Jacobian.
        ee_position = estimate_data.site_xpos[ee_site_id].copy()
        ee_rotation = estimate_data.site_xmat[ee_site_id].reshape(3, 3).copy()
        return ee_position, ee_rotation, jacp[:, v_ids]

    # ---- PBVS controller
    # Yara: Position-based visual servoing.
    # Cartesian error: e = target - end_effector
    # Desired velocity: v = PBVS_GAIN*e
    # DLS: qdot = J.T*(J*J.T + damping^2*I)^-1*v
    def pbvs_controller(target_position, x_hat):
        ee_position, _, J = get_estimated_ee_state(x_hat[:6])

        # Cartesian position error [m].
        position_error = target_position - ee_position

        # Damped least-squares: convert Cartesian error into joint velocities.
        A = J @ J.T + DAMPING**2 * np.eye(3)
        qdot = J.T @ np.linalg.solve(A, PBVS_GAIN * position_error)

        # Limit joint speeds to +/-0.35 rad/s.
        return np.clip(qdot, -MAX_JOINT_SPEED, MAX_JOINT_SPEED), position_error

    # ---- LQR joint controller
    # Yara: LQR and estimated-state feedback.
    # u = q_desired + K_lqr*(x_desired - x_hat)
    def lqr_controller(q_desired, qdot_desired, x_hat):
        x_desired = np.concatenate([q_desired, qdot_desired])
        u_raw = q_desired + K_lqr @ (x_desired - x_hat)

        # Apply actuator limits.
        return np.clip(u_raw, lower, upper), u_raw

    # ---- Simulated grasp
    # Robot task integration.
    # Cube is attached mathematically, not by physical gripper contact.
    cube_attached = False
    cube_local_offset = None
    cube_saved_quat = None

    # Save original cube collision settings.
    original_contype = int(model.geom_contype[cube_geom_id])
    original_conaffinity = int(model.geom_conaffinity[cube_geom_id])

    # Return end-effector position and rotation.
    def ee_pose():
        return (data.site_xpos[ee_site_id].copy(),
                data.site_xmat[ee_site_id].reshape(3, 3).copy())

    # ---- Attach cube
    # Save cube position relative to the end effector.
    # Local offset = R.T*(cube_position - ee_position)
    def attach_cube():
        nonlocal cube_attached, cube_local_offset, cube_saved_quat

        mujoco.mj_forward(model, data)
        ee_pos, ee_rot = ee_pose()

        cube_pos = data.qpos[cube_qadr:cube_qadr + 3].copy()
        cube_saved_quat = data.qpos[cube_qadr + 3:cube_qadr + 7].copy()   # Orientation as a quaternion.
        cube_local_offset = ee_rot.T @ (cube_pos - ee_pos)                # Offset in end-effector coordinates.

        # Disable cube collisions during transport.
        model.geom_contype[cube_geom_id] = 0
        model.geom_conaffinity[cube_geom_id] = 0

        cube_attached = True
        print("Cube attached (simulated grasp)")

    # ---- Carry attached cube
    # Keep cube at a fixed offset from the moving end effector.
    # cube_position = ee_position + R*local_offset
    def update_attached_cube():
        if not cube_attached:
            return

        ee_pos, ee_rot = ee_pose()
        data.qpos[cube_qadr:cube_qadr + 3] = ee_pos + ee_rot @ cube_local_offset  # Update position.
        data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_saved_quat                  # Preserve orientation.
        data.qvel[cube_vadr:cube_vadr + 6] = 0.0                                  # Zero velocity while attached.
        mujoco.mj_forward(model, data)

    # ---- Release cube
    # Detach cube and restore normal collision behaviour.
    def release_cube():
        nonlocal cube_attached
        cube_attached = False
        model.geom_contype[cube_geom_id] = original_contype
        model.geom_conaffinity[cube_geom_id] = original_conaffinity
        mujoco.mj_forward(model, data)
        print("Cube released")

    # ---- Placement target
    # Account for the cube offset from the end effector.
    # ee_target = cube_drop_position - R*local_offset
    def get_place_target(x_hat):
        _, ee_rot, _ = get_estimated_ee_state(x_hat[:6])
        return drop_cube_position - ee_rot @ cube_local_offset

    # ---- Initialise state machine
    stage = "DETECT"
    cube_target = None
    detection_count = 0
    q_desired = view_q.copy()

    next_perception = 0.0
    next_print = 0.0
    next_log_time = 0.0

    stable_count = 0
    home_stable_count = 0
    clip_count = 0

    completed = False
    settle_start_time = None

    performance_data = []
    stage_times = {"DETECT": 0.0}

    # ---- Change stage
    # Move to the next task stage and reset stability counters.
    def change_stage(new_stage):
        nonlocal stage, stable_count, home_stable_count
        stage = new_stage
        stable_count = 0
        home_stable_count = 0
        stage_times[new_stage] = float(data.time)
        print("\nSTAGE:", stage)

    # ---- Check target reached
    # Require position error below tolerance for 20 consecutive steps.
    def target_reached(error, tolerance):
        nonlocal stable_count
        stable_count = stable_count + 1 if np.linalg.norm(error) < tolerance else 0
        return stable_count >= STABLE_STEPS

    # ---- Log performance
    # Record desired, actual and estimated states for Step 5.
    # PBVS error is recorded in metres.
    def record_performance(x_hat, qdot_desired, position_error):
        row = {
            "time_s": float(data.time),
            "stage": stage,
            "pbvs_error_m": (float(np.linalg.norm(position_error))
                             if position_error is not None else np.nan),
        }
        for i in range(6):
            j = i + 1
            # Joint positions [rad].
            row[f"q_desired_{j}"] = float(q_desired[i])
            row[f"q_actual_{j}"] = float(data.qpos[q_ids[i]])
            row[f"q_estimated_{j}"] = float(x_hat[i])
            # Joint velocities [rad/s].
            row[f"dq_desired_{j}"] = float(qdot_desired[i])
            row[f"dq_actual_{j}"] = float(data.qvel[v_ids[i]])
            row[f"dq_estimated_{j}"] = float(x_hat[i + 6])
            # Actuator position command [rad].
            row[f"control_{j}"] = float(data.ctrl[actuator_ids[i]])
        performance_data.append(row)

    # ---- Main loop: run perception, Kalman estimation, PBVS and LQR together.
    print("\nUR3e vision-guided pick-and-place started")
    print("STAGE:", stage)

    # Open the viewer only when show_viewer=True.
    viewer_context = (mujoco.viewer.launch_passive if show_viewer
                      else lambda *_: contextlib.nullcontext())

    try:
        with viewer_context(model, data) as viewer:
            while data.time < MAX_TIME:

                # Stop if the user closes the viewer.
                if viewer is not None and not viewer.is_running():
                    break

                step_start = time.time()

                # ---- STAGE 1: Detect cube
                # Aalia: RGB-D object detection.
                # Require three consecutive successful measurements.
                if stage == "DETECT" and data.time >= next_perception:
                    next_perception += PERCEPTION_PERIOD
                    measurement = perception.measure(data)
                    target, _ = target_filter.update(measurement)
                    detection_count = detection_count + 1 if measurement is not None else 0

                    if target is not None and detection_count >= REQUIRED_DETECTIONS:
                        cube_target = target.copy()
                        print("Cube detected:", np.round(cube_target, 3))
                        change_stage("APPROACH")

                # ---- Update Kalman filter
                # Sameeha: Estimate robot joint states.
                x_hat, _ = observer.update(data.sensordata, data.ctrl)

                # ---- Cartesian target
                # Each stage has a different target position.
                target_position = None

                if stage in HEIGHT_ABOVE_CUBE:
                    # APPROACH, LOWER and LIFT use the detected cube position.
                    target_position = cube_target.copy()
                    target_position[2] += HEIGHT_ABOVE_CUBE[stage]
                elif stage == "TRANSPORT":
                    # TRANSPORT moves the cube above the drop zone.
                    target_position = np.array([DROP_XY[0], DROP_XY[1], cube_target[2] + LIFT_HEIGHT])
                elif stage == "PLACE":
                    # PLACE accounts for the cube-to-tool offset.
                    target_position = get_place_target(x_hat)
                elif stage == "RETRACT":
                    # RETRACT raises the arm after releasing the cube.
                    target_position = np.array([DROP_XY[0], DROP_XY[1],
                                                drop_cube_position[2] + RETRACT_HEIGHT])

                # Reset desired velocity for each control step.
                qdot_desired = np.zeros(6)
                position_error = None
                status = None

                # ---- STAGE 8: Return home
                # Return to the saved viewing configuration.
                # q_next = q_desired + limited_joint_step
                if stage == "RETURN_HOME":
                    # Joint error relative to home.
                    joint_error = home_q - x_hat[:6]

                    # Gradually move the desired joints towards home, limited per timestep.
                    max_step = HOME_MAX_SPEED * dt
                    joint_step = np.clip(home_q - q_desired, -max_step, max_step)
                    q_desired = np.clip(q_desired + joint_step, lower, upper)
                    qdot_desired = joint_step / dt

                    # Check joint error and actual joint velocity; require 20 consecutive stable steps.
                    at_home = (np.all(np.abs(joint_error) < HOME_TOLERANCE)
                               and np.all(np.abs(data.qvel[v_ids]) < 0.05))
                    home_stable_count = home_stable_count + 1 if at_home else 0

                    if home_stable_count >= HOME_STABLE_STEPS:
                        q_desired = home_q.copy()
                        qdot_desired = np.zeros(6)
                        print("Robot returned to starting position")
                        settle_start_time = float(data.time)
                        change_stage("SETTLE")

                    status = f"joint error={np.max(np.abs(joint_error)):.4f}rad"

                # ---- STAGES 2-7: PBVS movement
                # Move towards the current Cartesian target.
                elif target_position is not None:
                    qdot_command, position_error = pbvs_controller(target_position, x_hat)

                    # Use tighter tolerance during placement.
                    tolerance = PLACE_TOLERANCE if stage == "PLACE" else POSITION_TOLERANCE

                    # Check if the stage is complete.
                    if target_reached(position_error, tolerance):
                        if stage == "LOWER":
                            # LOWER: attach the cube.
                            attach_cube()
                        elif stage == "PLACE":
                            # PLACE: update cube pose and release it.
                            update_attached_cube()
                            print("Cube placement position:",
                                  np.round(data.qpos[cube_qadr:cube_qadr + 3], 4))
                            release_cube()

                        # Continue to the next task stage.
                        change_stage(NEXT_STAGE[stage])
                    else:
                        # Update desired joints: q_next = q_current + qdot*dt
                        qdot_desired = qdot_command
                        q_desired = np.clip(q_desired + qdot_desired * dt, lower, upper)

                    status = f"error={np.linalg.norm(position_error):.4f}m"

                # Display the active stage and error once per second.
                if status is not None and data.time >= next_print:
                    next_print += 1.0
                    print(f"t={data.time:.1f}s | stage={stage} | {status}")

                # ---- STAGE 9: Settle
                # Wait five seconds after returning home.
                if stage == "SETTLE" and data.time - settle_start_time >= SETTLE_TIME:
                    completed = True
                    change_stage("DONE")

                # ---- LQR joint tracking
                # Yara: Estimated-state feedback.
                # Track desired joint positions and velocities.
                u, u_raw = lqr_controller(q_desired, qdot_desired, x_hat)

                # Count timesteps with actuator command clipping.
                if np.any(np.abs(u_raw - u) > 1e-8):
                    clip_count += 1

                # Send position commands to UR3e actuators.
                data.ctrl[actuator_ids] = u

                # ---- Save performance data for Step 5 when logging is enabled.
                if log_performance and data.time >= next_log_time:
                    record_performance(x_hat, qdot_desired, position_error)
                    next_log_time += LOG_PERIOD

                mujoco.mj_step(model, data)

                # Move the cube with the end effector while attached.
                update_attached_cube()

                if viewer is not None:
                    viewer.sync()

                # End simulation once all stages are completed.
                if stage == "DONE":
                    break

                # Pace the viewer to real time.
                if viewer is not None:
                    remaining = dt - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        # Release camera rendering resources.
        perception.close()

    # ---- Final results
    # Placement error = distance between final and desired cube positions.
    # Home error = maximum absolute joint-angle difference.
    final_cube_position = data.qpos[cube_qadr:cube_qadr + 3].copy()
    placement_error = float(np.linalg.norm(final_cube_position - drop_cube_position))
    final_home_error = float(np.max(np.abs(data.qpos[q_ids] - home_q)))

    print("\nSTEP 4 RESULTS")
    print("Final stage:", stage)
    print("Pick-and-place completed:", completed)
    print("Actuator clipping steps:", clip_count)
    print("Final cube position:", np.round(final_cube_position, 4))
    print("Target cube position:", np.round(drop_cube_position, 4))
    print(f"Final placement error: {placement_error:.4f} m")
    print(f"Final home joint error: {final_home_error:.4f} rad")
    print("Simulation finished")

    # Used by main.py
    return {
        "completed": completed,
        "final_stage": stage,
        "cube_seed": cube_seed,
        "task_duration_s": float(data.time),
        "clipping_steps": clip_count,
        "placement_error_m": placement_error,
        "max_lqr_eigenvalue": max_eigenvalue,
        "final_cube_position": final_cube_position,
        "target_cube_position": drop_cube_position,
        "final_home_error_rad": final_home_error,
        "stage_times": stage_times,
        "performance_data": performance_data,
    }


if __name__ == "__main__":
    run_simulation()