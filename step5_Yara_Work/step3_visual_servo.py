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

APPROACH_HEIGHT = 0.15
POSITION_TOLERANCE = 0.01

PERCEPTION_PERIOD = 0.1
MAX_TIME = 30.0
REQUIRED_DETECTIONS = 3

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
        sensor.name
        for sensor in spec.sensors
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

lower = model.actuator_ctrlrange[
    actuator_ids, 0
]

upper = model.actuator_ctrlrange[
    actuator_ids, 1
]



perception = ObjectPerception(model, seed=1)
target_filter = TargetFilter()

pose_arm_to_look_down(
    model,
    data,
    perception.cam.cam_id
)

# Initialise controller at viewing pose
view_q = data.qpos[q_ids].copy()

data.ctrl[actuator_ids] = view_q

mujoco.mj_forward(model, data)

observer = JointObserver(
    model,
    q0=view_q,
    seed=0
)

print("Kalman filter initialised")
print("Encoder sensor indices:", observer.sensor_adr)

arm = Arm(model)

# Linearise around the viewing pose
Ad, Bd, x_star, u_star = linearise(
    arm,
    view_q
)

# Penalise joint position and velocity errors
Q = np.diag(
    [Q_POSITION] * 6 +
    [Q_VELOCITY] * 6
)

# Penalise actuator effort
R = np.diag([R_CONTROL] * 6)

# Solve discrete Riccati equation
P_lqr = solve_discrete_are(
    Ad,
    Bd,
    Q,
    R
)

# Calculate LQR gain
K_lqr = np.linalg.solve(
    R + Bd.T @ P_lqr @ Bd,
    Bd.T @ P_lqr @ Ad
)

# Check linearised closed-loop stability
closed_loop = Ad - Bd @ K_lqr

max_eigenvalue = np.max(
    np.abs(np.linalg.eigvals(closed_loop))
)

print("LQR maximum eigenvalue:", max_eigenvalue)

if max_eigenvalue >= 1.0:
    raise RuntimeError(
        "LQR linearised closed-loop system is unstable"
    )


estimate_data = mujoco.MjData(model)


def get_estimated_ee_state(q_hat):

    estimate_data.qpos[:] = data.qpos

    # Replace robot angles with estimated values
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

    state_error = (
        x_desired - x_hat
    )

    u = (
        q_desired +
        K_lqr @ state_error
    )

    return u


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


            if data.time >= next_perception:

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

            x_hat, P = observer.update(
                data.sensordata,
                data.ctrl
            )


            qdot_desired[:] = 0.0

            if (
                target is not None
                and detection_count >= REQUIRED_DETECTIONS
            ):

                # Move above the detected cube
                target_position = target.copy()

                target_position[2] += (
                    APPROACH_HEIGHT
                )

                qdot_command, position_error = (
                    pbvs_controller(
                        target_position,
                        x_hat
                    )
                )

                error_norm = np.linalg.norm(
                    position_error
                )

                reached_target = (
                    error_norm < POSITION_TOLERANCE
                )

                if not reached_target:

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
                        f"error={error_norm:.4f}m | "
                        f"target={np.round(target_position, 3)}"
                    )


            u = lqr_controller(
                q_desired,
                qdot_desired,
                x_hat
            )

            # Enforce actuator limits
            u_before_clip = u.copy()

            u = np.clip(
                u,
                lower,
                upper
            )

            if np.any(
                np.abs(u_before_clip - u) > 1e-8
            ):
                clip_count += 1

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

    perception.close()



print("\nPBVS simulation finished")

print("Target reached:", reached_target)

print("Actuator clipping steps:", clip_count)

print(
    "Final estimated joint positions:",
    np.round(x_hat[:6], 4)
)