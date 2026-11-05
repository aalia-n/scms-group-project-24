# STEP 6: FINAL SYSTEM INTEGRATION
# This code runs the complete UR3e vision-guided pick-and-place system by calling the Step 4 simulation.
# The integrated system combines Aalia's RGB-D camera perception, Sameeha's Kalman state estimation,
# Yara's PBVS and LQR controllers, and the MuJoCo robot simulation.
# The robot detects a cube, picks it up, transports it, places it at the target and returns to its starting pose.
# The code displays final task completion and performance results.

# ---- Import integrated simulation
# Step 4 already combines perception, Kalman, PBVS and LQR.
from step5_Yara_Work.step4_pick_and_place import run_simulation

# ---- Simulation settings
# Choose a reproducible cube starting position.
# Enable the MuJoCo viewer to observe the full task.
CUBE_SEED = 5
SHOW_VIEWER = True


# ---- Final integration function
# Run the complete robot system and report its results.
def run_final_system():

    # ---- Display system information
    # Identify the integrated team components.
    print("\n ---- UR3e FINAL INTEGRATED SYSTEM")
    print("\nIntegrated components:")
    print("1. Aalia   - RGB-D perception")
    print("2. Sameeha - Kalman state estimation")
    print("3. Yara    - PBVS and LQR control")
    print("4. MuJoCo  - Pick-and-place execution")
    print("\nStarting integrated simulation...\n")

    # ---- Run complete pick-and-place
    # Perception, estimation and control integration.
    # Calls the full state machine implemented in Step 4.
    # Performance logging is disabled for the final demonstration.
    results = run_simulation(log_performance=False, cube_seed=CUBE_SEED, show_viewer=SHOW_VIEWER)

    # ---- Display final results
    # Evaluate whether the complete robot task was successful.
    print("\nFINAL INTEGRATION RESULTS")

    # Task completion: True means the state machine reached DONE.
    print("Task completed:", results["completed"])

    # Final task stage: the last stage reached by the robot.
    print("Final stage:", results["final_stage"])

    # Simulation duration: total simulated time taken by the task [s].
    print("Simulation duration:", round(results["task_duration_s"], 3), "seconds")

    # Final placement error.
    # (Cartesian positioning accuracy)
    # Error = ||final_cube_position - target_cube_position||, converted from metres to millimetres.
    print("Final placement error:", round(results["placement_error_m"] * 1000, 3), "mm")

    # Actuator clipping.
    # (control input constraints)
    # Count timesteps when commands exceeded actuator limits.
    print("Actuator clipping steps:", results["clipping_steps"])

    # LQR closed-loop stability.
    # (State-feedback stability)
    # Discrete-time linearised stability requires |eigenvalue| < 1.
    print("Maximum LQR eigenvalue magnitude:", results["max_lqr_eigenvalue"])

    # ---- Check final task success
    # Display success only when all required stages are completed.
    print("\nFINAL SYSTEM:", "TASK COMPLETED" if results["completed"] else "TASK NOT COMPLETED")

    return results

if __name__ == "__main__":
    run_final_system()