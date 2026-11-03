
"""
  main()    runs the complete UR3e vision-guided pick-and-place system

The integrated system uses:
  Aalia     wrist RGB-D camera, object detection and target filtering
  Sameeha   state-space model and Kalman observer
  Yara      PBVS, LQR joint control and pick-and-place execution

The complete simulation loop is implemented in step5_Yara_Work/step4_pick_and_place.py.
This file launches the integrated system so that all team members run the same pipeline.

main.py   -> opens the viewer, runs pick-and-place and prints the final results
"""

from step5_Yara_Work.step4_pick_and_place import run_simulation  # Yara: integrated controller (Aalia + Sameeha + Yara)


# ---------------------------------------------------------------- simulation settings
CUBE_SEED = 1          # change this to test different randomised object positions
SHOW_VIEWER = True     # open the MuJoCo viewer during the task


def main(seed=None):
    if seed is None:
        seed = CUBE_SEED

    print("\nUR3e TEAM INTEGRATED SYSTEM")

    print("\nTeam contributions:")
    print("Aalia   - RGB-D camera and object perception")
    print("Sameeha - Kalman state estimation")
    print("Yara    - PBVS and LQR control")

    print("\nStarting simulation...")
    print("Cube seed:", seed)

    # ---- integrated simulation (Aalia + Sameeha + Yara)
    # Builds the MuJoCo scene, detects the object, estimates the joint states,
    # computes PBVS targets and LQR commands, then executes pick-and-place.
    results = run_simulation(
        log_performance=False,
        cube_seed=seed,
        show_viewer=SHOW_VIEWER
    )

    # ---- final results
    print("\nTEAM INTEGRATION RESULTS")

    print("Task completed:", results["completed"])
    print("Final stage:", results["final_stage"])
    print(f"Task duration: {results['task_duration_s']:.2f} s")

    if results["placement_error_m"] is not None:
        print(f"Final placement error: {results['placement_error_m'] * 1000:.3f} mm")

    print("Actuator clipping steps:", results["clipping_steps"])

    # ---- task completion
    if results["completed"]:
        print("\nFINAL RESULT: TASK COMPLETED")
    else:
        print("\nFINAL RESULT: TASK NOT COMPLETED")

    return results


if __name__ == "__main__":
    main()
