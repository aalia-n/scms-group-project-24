import time
import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env

#LOAD SHARED ENVIRONMENT
model = env.build_scene()
data = mujoco.MjData(model)

# Reset the robot to its home configuration
env.reset_home(model, data)


#DEFINE ROBOT JOINT POSITIONS

# Starting position from the XML home keyframe
home_q = np.array([
    -1.5708,
    -1.5708,
     1.5708,
    -1.5708,
    -1.5708,
     0.0
])

# Test target position
target_q = np.array([
    -1.0,
    -1.2,
     1.2,
    -1.0,
    -1.2,
     0.5
])

#MOVEMENT

# Time to move between positions (seconds)
MOVE_TIME = 4.0

# Time to hold each position (seconds)
HOLD_TIME = 2.0

# Total cycle time
CYCLE_TIME = 2 * MOVE_TIME + 2 * HOLD_TIME

# STEP 4: SMOOTH TRAJECTORY

def smooth_move(start_q, end_q, progress):
    """
    Generate a smooth joint trajectory.

    progress = 0 means starting position
    progress = 1 means target position

    Smoothstep gives zero commanded velocity
    at the start and end of each movement.
    """

    progress = np.clip(progress, 0.0, 1.0)

    # Cubic smoothstep
    s = 3 * progress**2 - 2 * progress**3

    # Interpolate between the joint positions
    q_desired = start_q + s * (end_q - start_q)

    return q_desired

# STEP 5: MAIN SIMULATION

print("\nUR3e automatic joint control started")

print("Home position:", home_q)
print("Target position:", target_q)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        step_start = time.time()

        # Current time within the movement cycle
        t = data.time % CYCLE_TIME

        # from home to target

        if t < MOVE_TIME:

            progress = t / MOVE_TIME

            q_desired = smooth_move(
                home_q,
                target_q,
                progress
            )

        elif t < MOVE_TIME + HOLD_TIME:

            q_desired = target_q.copy()

        # from target to home

        elif t < 2 * MOVE_TIME + HOLD_TIME:

            progress = (
                t - MOVE_TIME - HOLD_TIME
            ) / MOVE_TIME

            q_desired = smooth_move(
                target_q,
                home_q,
                progress
            )

        else:

            q_desired = home_q.copy()

        # SEND POSITION COMMANDS TO ROBOT


        # The six actuators accept desired
        # joint angles in radians.
        data.ctrl[:6] = q_desired

        # Run one MuJoCo physics step
        mujoco.mj_step(model, data)

        # Update the viewer
        viewer.sync()

        # Keep simulation near real-time
        elapsed = time.time() - step_start

        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)
