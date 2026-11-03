from step5_Yara_Work.step4_pick_and_place import run_simulation

CUBE_SEED = 5

SHOW_VIEWER = True

def run_final_system():

    print("\n====================================")
    print("UR3e FINAL INTEGRATED SYSTEM")
    print("====================================")

    print("\nIntegrated components:")
    print("1. Aalia   - RGB-D perception")
    print("2. Sameeha - Kalman state estimation")
    print("3. Yara    - PBVS and LQR control")
    print("4. MuJoCo  - Pick-and-place execution")

    print("\nStarting integrated simulation...\n")

    results = run_simulation(
        log_performance=False,
        cube_seed=CUBE_SEED,
        show_viewer=SHOW_VIEWER
    )

    print("\n====================================")
    print("FINAL INTEGRATION RESULTS")
    print("====================================")

    print(
        "Task completed:",
        results["completed"]
    )

    print(
        "Final stage:",
        results["final_stage"]
    )

    print(
        "Simulation duration:",
        round(results["task_duration_s"], 3),
        "seconds"
    )

    print(
        "Final placement error:",
        round(
            results["placement_error_m"] * 1000,
            3
        ),
        "mm"
    )

    print(
        "Actuator clipping steps:",
        results["clipping_steps"]
    )

    print(
        "Maximum LQR eigenvalue magnitude:",
        results["max_lqr_eigenvalue"]
    )

    if results["completed"]:

        print("\nFINAL SYSTEM: TASK COMPLETED")

    else:

        print("\nFINAL SYSTEM: TASK NOT COMPLETED")

    return results


if __name__ == "__main__":
    run_final_system()
