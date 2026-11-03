import contextlib
import time

import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env
from step3_Aalia_Work.step6_perception import ObjectPerception, TargetFilter
from step3_Aalia_Work.step1_camera import pose_arm_to_look_down
from step4_Sameeha_Work.observer import JointObserver, add_encoder_sensors
from step4_Sameeha_Work.step1_state_space_model import Arm, linearise
from scipy.linalg import solve_discrete_are


# ---------------------------------------------------------------- robot joints
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
                  "wrist_1", "wrist_2", "wrist_3"]

# ---------------------------------------------------------------- PBVS settings
PBVS_GAIN = 0.8
DAMPING = 0.05
MAX_JOINT_SPEED = 0.35
POSITION_TOLERANCE = 0.012
PLACE_TOLERANCE = 0.002
PLACE_CLEARANCE = 0.0005

# ---------------------------------------------------------------- perception settings
PERCEPTION_PERIOD = 0.1
REQUIRED_DETECTIONS = 3

# ---------------------------------------------------------------- simulation settings
MAX_TIME = 25.0
SETTLE_TIME = 5.0
STABLE_STEPS = 20

# ---------------------------------------------------------------- pick-and-place settings
APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.035
LIFT_HEIGHT = 0.18
DROP_XY = np.array([0.25, -0.20])

# ---------------------------------------------------------------- return-home settings
RETRACT_HEIGHT = 0.15
HOME_TOLERANCE = 0.02
HOME_STABLE_STEPS = 20
HOME_MAX_SPEED = 0.35

# ---------------------------------------------------------------- LQR settings
Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0

# ---------------------------------------------------------------- performance logging
LOG_PERIOD = 0.05

# ---------------------------------------------------------------- state machine
# Stage to move to once the PBVS target for the current stage is reached.
NEXT_STAGE = {
    "APPROACH": "LOWER",
    "LOWER": "LIFT",
    "LIFT": "TRANSPORT",
    "TRANSPORT": "PLACE",
    "PLACE": "RETRACT",
    "RETRACT": "RETURN_HOME",
}

# Stages whose target is directly above the detected cube, and at what height.
HEIGHT_ABOVE_CUBE = {
    "APPROACH": APPROACH_HEIGHT,
    "LOWER": GRASP_HEIGHT,
    "LIFT": LIFT_HEIGHT,
}


def build_model():
    """Build the scene, adding joint encoder sensors at compile time if missing."""
    original_compile = mujoco.MjSpec.compile
    encoder_names = ["enc_" + name for name in JOINT_NAMES]

    def compile_with_encoders(spec):
        existing = {sensor.name for sensor in spec.sensors}
        present = [name in existing for name in encoder_names]
        if not all(present):
            if any(present):
                raise RuntimeError("Only some encoder sensors exist.")
            add_encoder_sensors(spec)
        return original_compile(spec)

    mujoco.MjSpec.compile = compile_with_encoders
    try:
        return env.build_scene()
    finally:
        mujoco.MjSpec.compile = original_compile


def run_simulation(log_performance=False, cube_seed=1, show_viewer=True):

    print("\nSTEP 4 - PICK AND PLACE")

    model = build_model()
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    env.reset_home(model, data)
    env.randomise_object(model, data, np.random.default_rng(cube_seed))

    q_ids = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES])
    v_ids = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES])
    actuator_ids = np.array([model.actuator(n).id for n in ACTUATOR_NAMES])
    ee_site_id = model.site("attachment_site").id

    cube_joint = model.joint("target_free")
    cube_qadr = cube_joint.qposadr[0]
    cube_vadr = cube_joint.dofadr[0]
    cube_geom_id = model.geom("target_geom").id

    lower = model.actuator_ctrlrange[actuator_ids, 0]
    upper = model.actuator_ctrlrange[actuator_ids, 1]

    drop_cube_position = np.array([
        DROP_XY[0], DROP_XY[1],
        env.TABLE_TOP_Z + env.OBJ_HALF + PLACE_CLEARANCE,
    ])

    # ---- camera and perception (Aalia)
    perception = ObjectPerception(model, seed=cube_seed)
    target_filter = TargetFilter()
    pose_arm_to_look_down(model, data, perception.cam.cam_id)

    view_q = data.qpos[q_ids].copy()
    data.ctrl[actuator_ids] = view_q
    mujoco.mj_forward(model, data)

    # The robot returns to this viewing configuration after placing the cube.
    home_q = view_q.copy()

    # ---- Kalman observer and linearised model (Sameeha)
    observer = JointObserver(model, q0=view_q, seed=0)
    print("Kalman filter initialised")

    Ad, Bd, _, _ = linearise(Arm(model), view_q)

    # ---- LQR design (Yara)
    Q = np.diag([Q_POSITION] * 6 + [Q_VELOCITY] * 6)
    R = np.diag([R_CONTROL] * 6)
    P_lqr = solve_discrete_are(Ad, Bd, Q, R)
    K_lqr = np.linalg.solve(R + Bd.T @ P_lqr @ Bd, Bd.T @ P_lqr @ Ad)

    max_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K_lqr))))
    print("LQR maximum eigenvalue:", max_eigenvalue)
    if max_eigenvalue >= 1.0:
        raise RuntimeError("Linearised LQR system is unstable")

    estimate_data = mujoco.MjData(model)

    def get_estimated_ee_state(q_hat):
        """End-effector position, rotation and position Jacobian at the estimated joint angles."""
        estimate_data.qpos[:] = data.qpos
        estimate_data.qpos[q_ids] = q_hat
        estimate_data.qvel[:] = 0.0
        mujoco.mj_forward(model, estimate_data)

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, estimate_data, jacp, None, ee_site_id)

        ee_position = estimate_data.site_xpos[ee_site_id].copy()
        ee_rotation = estimate_data.site_xmat[ee_site_id].reshape(3, 3).copy()
        return ee_position, ee_rotation, jacp[:, v_ids]

    # ---- position-based visual servoing (Yara)
    def pbvs_controller(target_position, x_hat):
        ee_position, _, J = get_estimated_ee_state(x_hat[:6])
        position_error = target_position - ee_position

        # Damped least-squares inverse kinematics.
        A = J @ J.T + DAMPING**2 * np.eye(3)
        qdot = J.T @ np.linalg.solve(A, PBVS_GAIN * position_error)

        return np.clip(qdot, -MAX_JOINT_SPEED, MAX_JOINT_SPEED), position_error

    # ---- LQR joint tracking (Yara)
    def lqr_controller(q_desired, qdot_desired, x_hat):
        x_desired = np.concatenate([q_desired, qdot_desired])
        u_raw = q_desired + K_lqr @ (x_desired - x_hat)
        return np.clip(u_raw, lower, upper), u_raw

    # ---- simulated grasp
    cube_attached = False
    cube_local_offset = None
    cube_saved_quat = None
    original_contype = int(model.geom_contype[cube_geom_id])
    original_conaffinity = int(model.geom_conaffinity[cube_geom_id])

    def ee_pose():
        return (data.site_xpos[ee_site_id].copy(),
                data.site_xmat[ee_site_id].reshape(3, 3).copy())

    def attach_cube():
        nonlocal cube_attached, cube_local_offset, cube_saved_quat

        mujoco.mj_forward(model, data)
        ee_pos, ee_rot = ee_pose()

        cube_pos = data.qpos[cube_qadr:cube_qadr + 3].copy()
        cube_saved_quat = data.qpos[cube_qadr + 3:cube_qadr + 7].copy()
        cube_local_offset = ee_rot.T @ (cube_pos - ee_pos)

        # Disable cube collisions while it is carried.
        model.geom_contype[cube_geom_id] = 0
        model.geom_conaffinity[cube_geom_id] = 0

        cube_attached = True
        print("Cube attached (simulated grasp)")

    def update_attached_cube():
        if not cube_attached:
            return
        ee_pos, ee_rot = ee_pose()
        data.qpos[cube_qadr:cube_qadr + 3] = ee_pos + ee_rot @ cube_local_offset
        data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_saved_quat
        data.qvel[cube_vadr:cube_vadr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def release_cube():
        # Called straight after update_attached_cube(), so the cube is already
        # at its final pose with zero velocity.
        nonlocal cube_attached
        cube_attached = False

        # Restore cube collisions after release.
        model.geom_contype[cube_geom_id] = original_contype
        model.geom_conaffinity[cube_geom_id] = original_conaffinity
        mujoco.mj_forward(model, data)

        print("Cube released")

    def get_place_target(x_hat):
        _, ee_rot, _ = get_estimated_ee_state(x_hat[:6])
        return drop_cube_position - ee_rot @ cube_local_offset

    # ---- pick-and-place state machine
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

    def change_stage(new_stage):
        nonlocal stage, stable_count, home_stable_count
        stage = new_stage
        stable_count = 0
        home_stable_count = 0
        stage_times[new_stage] = float(data.time)
        print("\nSTAGE:", stage)

    def target_reached(error, tolerance):
        nonlocal stable_count
        stable_count = stable_count + 1 if np.linalg.norm(error) < tolerance else 0
        return stable_count >= STABLE_STEPS

    def record_performance(x_hat, qdot_desired, position_error):
        row = {
            "time_s": float(data.time),
            "stage": stage,
            "pbvs_error_m": (float(np.linalg.norm(position_error))
                             if position_error is not None else np.nan),
        }
        for i in range(6):
            j = i + 1
            row[f"q_desired_{j}"] = float(q_desired[i])
            row[f"q_actual_{j}"] = float(data.qpos[q_ids[i]])
            row[f"q_estimated_{j}"] = float(x_hat[i])
            row[f"dq_desired_{j}"] = float(qdot_desired[i])
            row[f"dq_actual_{j}"] = float(data.qvel[v_ids[i]])
            row[f"dq_estimated_{j}"] = float(x_hat[i + 6])
            row[f"control_{j}"] = float(data.ctrl[actuator_ids[i]])
        performance_data.append(row)

    print("\nUR3e vision-guided pick-and-place started")
    print("STAGE:", stage)

    # ---- main simulation loop
    viewer_context = (mujoco.viewer.launch_passive if show_viewer
                      else lambda *_: contextlib.nullcontext())
    try:
        with viewer_context(model, data) as viewer:
            while data.time < MAX_TIME:

                if viewer is not None and not viewer.is_running():
                    break

                step_start = time.time()

                # ---- object detection (Aalia)
                if stage == "DETECT" and data.time >= next_perception:
                    next_perception += PERCEPTION_PERIOD
                    measurement = perception.measure(data)
                    target, _ = target_filter.update(measurement)
                    detection_count = detection_count + 1 if measurement is not None else 0

                    if target is not None and detection_count >= REQUIRED_DETECTIONS:
                        cube_target = target.copy()
                        print("Cube detected:", np.round(cube_target, 3))
                        change_stage("APPROACH")

                # ---- Kalman state estimation (Sameeha)
                x_hat, _ = observer.update(data.sensordata, data.ctrl)

                # ---- Cartesian target for the current stage
                target_position = None
                if stage in HEIGHT_ABOVE_CUBE:
                    target_position = cube_target.copy()
                    target_position[2] += HEIGHT_ABOVE_CUBE[stage]
                elif stage == "TRANSPORT":
                    target_position = np.array([DROP_XY[0], DROP_XY[1],
                                                cube_target[2] + LIFT_HEIGHT])
                elif stage == "PLACE":
                    target_position = get_place_target(x_hat)
                elif stage == "RETRACT":
                    target_position = np.array([DROP_XY[0], DROP_XY[1],
                                                drop_cube_position[2] + RETRACT_HEIGHT])

                qdot_desired = np.zeros(6)
                position_error = None
                status = None

                # ---- return to the original joint position
                if stage == "RETURN_HOME":
                    joint_error = home_q - x_hat[:6]

                    # Ramp the desired joints towards home at a limited speed
                    # rather than commanding an instantaneous jump.
                    max_step = HOME_MAX_SPEED * dt
                    joint_step = np.clip(home_q - q_desired, -max_step, max_step)
                    q_desired = np.clip(q_desired + joint_step, lower, upper)
                    qdot_desired = joint_step / dt

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

                # ---- PBVS movement control
                elif target_position is not None:
                    qdot_command, position_error = pbvs_controller(target_position, x_hat)
                    tolerance = PLACE_TOLERANCE if stage == "PLACE" else POSITION_TOLERANCE

                    if target_reached(position_error, tolerance):
                        if stage == "LOWER":
                            attach_cube()
                        elif stage == "PLACE":
                            update_attached_cube()
                            print("Cube placement position:",
                                  np.round(data.qpos[cube_qadr:cube_qadr + 3], 4))
                            release_cube()
                        change_stage(NEXT_STAGE[stage])
                    else:
                        qdot_desired = qdot_command
                        q_desired = np.clip(q_desired + qdot_desired * dt, lower, upper)

                    status = f"error={np.linalg.norm(position_error):.4f}m"

                if status is not None and data.time >= next_print:
                    next_print += 1.0
                    print(f"t={data.time:.1f}s | stage={stage} | {status}")

                # ---- let the cube settle after returning home
                if stage == "SETTLE" and data.time - settle_start_time >= SETTLE_TIME:
                    completed = True
                    change_stage("DONE")

                # ---- LQR control (Yara)
                u, u_raw = lqr_controller(q_desired, qdot_desired, x_hat)
                if np.any(np.abs(u_raw - u) > 1e-8):
                    clip_count += 1
                data.ctrl[actuator_ids] = u

                # ---- save performance data for Step 5
                if log_performance and data.time >= next_log_time:
                    record_performance(x_hat, qdot_desired, position_error)
                    next_log_time += LOG_PERIOD

                # ---- step simulation; carry the cube with the tool while attached
                mujoco.mj_step(model, data)
                update_attached_cube()

                if viewer is not None:
                    viewer.sync()

                if stage == "DONE":
                    break

                # ---- real-time viewer pacing
                if viewer is not None:
                    remaining = dt - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        perception.close()

    # ---- final placement results
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