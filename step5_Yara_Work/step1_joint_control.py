#this code jusy mainly focuses on testing the movemnts of the UR3e in the MUJOCO environemnt.

import time
import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env


# ---- Loading the Environemnt.
# Week 1: Dynamic systems
# Load the UR3e robot and initialise the MuJoCo simulation.

model = env.build_scene()
data = mujoco.MjData(model)

# Reset robot to its initial home position.
env.reset_home(model, data)


# ---- Defing the joint positions of the UR3e
# Week 1: Joint-space position control
# q = [q1, q2, q3, q4, q5, q6]
# Each value represents a joint angle in radians.

# Initial robot configuration.
home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# Desired target configuration.
target_q = np.array([-1.0, -1.2, 1.2, -1.0, -1.2, 0.5])


# ---- timing:
# Define how long the robot moves and holds each position.
# using simply maths to do so ----> Total cycle time = 2 * MOVE_TIME + 2 * HOLD_TIME

MOVE_TIME = 4.0
HOLD_TIME = 2.0

CYCLE_TIME = 2 * MOVE_TIME + 2 * HOLD_TIME


# create smooth trajectory 
# Week 1: Joint-space trajectory control
# Cubic smoothstep: s = 3p^2 - 2p^3
# Desired position: q_desired = q_start + s(q_end - q_start)
# Ensures zero commanded velocity at the start and end.

def smooth_move(start_q, end_q, progress):

    # Limit movement progress between 0 and 1.
    progress = np.clip(progress, 0.0, 1.0)

    # Calculate smooth interpolation factor.
    s = 3 * progress**2 - 2 * progress**3

    # Calculate desired joint positions.
    q_desired = start_q + s * (end_q - start_q)

    return q_desired


# ----  starting the sim 
# MuJoCo simulates the robot's motion and dynamics.
#printing comments to help with debugging.

print("\nUR3e automatic joint control started")

print("Home position:", home_q)
print("Target position:", target_q)

# ---- The main control loop (closed-loop control)
# Generate desired joint positions and send them to the actuators.

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        step_start = time.time()

        # ---- determine current movement stage
        # Modulo (%) repeats the 12-second movement cycle.
        t = data.time % CYCLE_TIME

        # ---- move from home to target (0-4 s)
        if t < MOVE_TIME:

            progress = t / MOVE_TIME

            q_desired = smooth_move(home_q, target_q, progress)

        # ---- hold target position (4-6 s)
        elif t < MOVE_TIME + HOLD_TIME:

            q_desired = target_q.copy()

        # ---- move from target to home (6-10 s)
        elif t < 2 * MOVE_TIME + HOLD_TIME:

            progress = (t - MOVE_TIME - HOLD_TIME) / MOVE_TIME

            q_desired = smooth_move(target_q, home_q, progress)

        # ---- hold home position (10-12 s)
        else:

            q_desired = home_q.copy()

        # ---- send joint position commands(closed loop control)
        # Position actuator feedback: error = q_desired - q_actual
        # MuJoCo's position actuators track the desired angles.

        data.ctrl[:6] = q_desired

        # ---- update robot dynamics
        # Advance the physics simulation by one timestep.

        mujoco.mj_step(model, data)

        # ---- update viewer
        viewer.sync()

        # ---- maintain real-time simulation
        # Synchronise execution with the MuJoCo timestep.

        elapsed = time.time() - step_start

        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)