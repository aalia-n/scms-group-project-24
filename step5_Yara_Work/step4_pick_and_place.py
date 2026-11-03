
import time
import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env

from step3_Aalia_Work.step6_perception import (
    ObjectPerception,
    TargetFilter
)

from step3_Aalia_Work.step1_camera import (
    pose_arm_to_look_down
)

from step4_Sameeha_Work.observer import (
    JointObserver,
    add_encoder_sensors
)

from step4_Sameeha_Work.step1_state_space_model import (
    Arm,
    linearise
)

from scipy.linalg import solve_discrete_are

JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

PBVS_GAIN = 0.8
DAMPING = 0.05
MAX_JOINT_SPEED = 0.35

POSITION_TOLERANCE = 0.012

PERCEPTION_PERIOD = 0.1
REQUIRED_DETECTIONS = 3

MAX_TIME = 45.0

APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.035
LIFT_HEIGHT = 0.18

DROP_XY = np.array([0.25, -0.20])

Q_POSITION = 500.0
Q_VELOCITY = 10.0
R_CONTROL = 1.0

original_compile = mujoco.MjSpec.compile


def compile_with_encoders(spec):

    encoder_names = [
        "enc_" + name
        for name in JOINT_NAMES
    ]

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
                "Only some encoder sensors exist."
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

env.randomise_object(
    model,
    data,
    np.random.default_rng(1)
)

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

ee_site_id = model.site("attachment_site").id

cube_joint = model.joint("target_free")

cube_qadr = cube_joint.qposadr[0]
cube_vadr = cube_joint.dofadr[0]

cube_geom_id = model.geom("target_geom").id

lower = model.actuator_ctrlrange[
    actuator_ids, 0
]

upper = model.actuator_ctrlrange[
    actuator_ids, 1
]


perception = ObjectPerception(model, seed=1)

target_filter = TargetFilter()

# Temporary camera viewing pose
pose_arm_to_look_down(
    model,
    data,
    perception.cam.cam_id
)

view_q = data.qpos[q_ids].copy()

data.ctrl[actuator_ids] = view_q

mujoco.mj_forward(model, data)


observer = JointObserver(
    model,
    q0=view_q,
    seed=0
)

print("Kalman filter initialised")

arm = Arm(model)

Ad, Bd, x_star, u_star = linearise(
    arm,
    view_q
)

Q = np.diag(
    [Q_POSITION] * 6 +
    [Q_VELOCITY] * 6
)

R = np.diag([R_CONTROL] * 6)

P_lqr = solve_discrete_are(
    Ad,
    Bd,
    Q,
    R
)

K_lqr = np.linalg.solve(
    R + Bd.T @ P_lqr @ Bd,
    Bd.T @ P_lqr @ Ad
)

max_eigenvalue = np.max(
    np.abs(
        np.linalg.eigvals(
            Ad - Bd @ K_lqr
        )
    )
)

print("LQR maximum eigenvalue:", max_eigenvalue)

if max_eigenvalue >= 1.0:
    raise RuntimeError("LQR model is unstable")


estimate_data = mujoco.MjData(model)


def get_estimated_ee_state(q_hat):

    estimate_data.qpos[:] = data.qpos

    estimate_data.qpos[q_ids] = q_hat

    estimate_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        estimate_data
    )

    ee_position = estimate_data.site_xpos[
        ee_site_id
    ].copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        estimate_data,
        jacp,
        jacr,
        ee_site_id
    )

    J = jacp[:, v_ids]

    return ee_position, J

def pbvs_controller(target_position, x_hat):

    q_hat = x_hat[:6]

    ee_position, J = get_estimated_ee_state(
        q_hat
    )

    position_error = (
        target_position - ee_position
    )

    v_desired = (
        PBVS_GAIN * position_error
    )

    A = (
        J @ J.T +
        DAMPING**2 * np.eye(3)
    )

    qdot_command = J.T @ np.linalg.solve(
        A,
        v_desired
    )

    qdot_command = np.clip(
        qdot_command,
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED
    )

    return qdot_command, position_error

def lqr_controller(q_desired, qdot_desired, x_hat):

    x_desired = np.concatenate([
        q_desired,
        qdot_desired
    ])

    state_error = x_desired - x_hat

    u = (
        q_desired +
        K_lqr @ state_error
    )

    return np.clip(
        u,
        lower,
        upper
    )


cube_attached = False

# Position of cube relative to tool frame
cube_local_offset = None

# Preserve cube orientation during carrying
cube_saved_quat = None

# Save original collision settings
original_contype = int(
    model.geom_contype[cube_geom_id]
)

original_conaffinity = int(
    model.geom_conaffinity[cube_geom_id]
)


def attach_cube():

    global cube_attached
    global cube_local_offset
    global cube_saved_quat

    mujoco.mj_forward(model, data)

    ee_pos = data.site_xpos[
        ee_site_id
    ].copy()

    ee_rot = data.site_xmat[
        ee_site_id
    ].reshape(3, 3).copy()

    cube_pos = data.qpos[
        cube_qadr:cube_qadr + 3
    ].copy()

    cube_saved_quat = data.qpos[
        cube_qadr + 3:cube_qadr + 7
    ].copy()

    # Cube offset expressed in tool coordinates
    cube_local_offset = (
        ee_rot.T @ (cube_pos - ee_pos)
    )

    # Disable cube contact while carried
    model.geom_contype[cube_geom_id] = 0
    model.geom_conaffinity[cube_geom_id] = 0

    cube_attached = True

    print("Cube attached (simulated grasp)")


def update_attached_cube():

    if not cube_attached:
        return

    ee_pos = data.site_xpos[
        ee_site_id
    ].copy()

    ee_rot = data.site_xmat[
        ee_site_id
    ].reshape(3, 3).copy()

    cube_pos = (
        ee_pos +
        ee_rot @ cube_local_offset
    )

    data.qpos[
        cube_qadr:cube_qadr + 3
    ] = cube_pos

    data.qpos[
        cube_qadr + 3:cube_qadr + 7
    ] = cube_saved_quat

    data.qvel[
        cube_vadr:cube_vadr + 6
    ] = 0.0

    mujoco.mj_forward(model, data)


def release_cube():

    global cube_attached

    # Ensure final position is updated
    update_attached_cube()

    cube_attached = False

    # Restore collision settings
    model.geom_contype[cube_geom_id] = (
        original_contype
    )

    model.geom_conaffinity[cube_geom_id] = (
        original_conaffinity
    )

    data.qvel[
        cube_vadr:cube_vadr + 6
    ] = 0.0

    mujoco.mj_forward(model, data)

    print("Cube released")

# PICK-AND-PLACE STATE MACHINE

stage = "DETECT"

cube_target = None

detection_count = 0

q_desired = view_q.copy()

next_perception = 0.0
next_print = 0.0

clip_count = 0

stable_count = 0

STABLE_STEPS = 20

completed = False


def change_stage(new_stage):

    global stage
    global stable_count

    stage = new_stage
    stable_count = 0

    print("\nSTAGE:", stage)


def target_reached(error):

    global stable_count

    if np.linalg.norm(error) < POSITION_TOLERANCE:
        stable_count += 1
    else:
        stable_count = 0

    return stable_count >= STABLE_STEPS


print("\nUR3e vision-guided pick-and-place started")

print("STAGE:", stage)


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

            if (
                stage == "DETECT"
                and data.time >= next_perception
            ):

                next_perception += PERCEPTION_PERIOD

                measurement = perception.measure(
                    data
                )

                target, target_cov = (
                    target_filter.update(
                        measurement
                    )
                )

                if measurement is not None:
                    detection_count += 1
                else:
                    detection_count = 0

                if (
                    target is not None
                    and detection_count >= REQUIRED_DETECTIONS
                ):
                    cube_target = target.copy()

                    print(
                        "Cube detected:",
                        np.round(cube_target, 3)
                    )

                    change_stage("APPROACH")

            x_hat, P = observer.update(
                data.sensordata,
                data.ctrl
            )

            target_position = None

            if stage == "APPROACH":

                target_position = cube_target.copy()

                target_position[2] += (
                    APPROACH_HEIGHT
                )

            elif stage == "LOWER":

                target_position = cube_target.copy()

                target_position[2] += (
                    GRASP_HEIGHT
                )

            elif stage == "LIFT":

                target_position = cube_target.copy()

                target_position[2] += (
                    LIFT_HEIGHT
                )

            elif stage == "TRANSPORT":

                target_position = np.array([
                    DROP_XY[0],
                    DROP_XY[1],
                    cube_target[2] + LIFT_HEIGHT
                ])

            qdot_desired = np.zeros(6)

            if target_position is not None:

                qdot_command, position_error = (
                    pbvs_controller(
                        target_position,
                        x_hat
                    )
                )

                error_norm = np.linalg.norm(
                    position_error
                )

                if target_reached(position_error):

                    if stage == "APPROACH":
                        change_stage("LOWER")

                    elif stage == "LOWER":
                        attach_cube()
                        change_stage("LIFT")

                    elif stage == "LIFT":
                        change_stage("TRANSPORT")

                    elif stage == "TRANSPORT":
                        release_cube()
                        change_stage("DONE")
                        completed = True

                else:

                    qdot_desired = qdot_command.copy()

                    q_desired += (
                        qdot_desired *
                        model.opt.timestep
                    )

                    q_desired = np.clip(
                        q_desired,
                        lower,
                        upper
                    )

                if data.time >= next_print:

                    next_print += 1.0

                    print(
                        f"t={data.time:.1f}s | "
                        f"stage={stage} | "
                        f"error={error_norm:.4f}m"
                    )

            x_desired = np.concatenate([
                q_desired,
                qdot_desired
            ])

            u_raw = (
                q_desired +
                K_lqr @ (x_desired - x_hat)
            )

            u = lqr_controller(
                q_desired,
                qdot_desired,
                x_hat
            )

            if np.any(
                np.abs(u_raw - u) > 1e-8
            ):
                clip_count += 1

            data.ctrl[actuator_ids] = u

            mujoco.mj_step(model, data)

            # Carry cube with the tool
            update_attached_cube()

            viewer.sync()

            if stage == "DONE":
                break

            elapsed = time.time() - step_start

            remaining = (
                model.opt.timestep - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

finally:

    perception.close()

print("\n========== STEP 4 RESULTS ==========")

print("Final stage:", stage)

print("Pick-and-place completed:", completed)

print("Actuator clipping steps:", clip_count)

print(
    "Final cube position:",
    np.round(
        data.qpos[
            cube_qadr:cube_qadr + 3
        ],
        4
    )
)

print("Simulation finished")