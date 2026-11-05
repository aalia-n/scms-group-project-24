# STEP 5: PERFORMANCE EVALUATION
# This code runs the complete Step 4 pick-and-place simulation and evaluates the performance of the integrated control system.
# It measures PBVS position error, LQR joint tracking accuracy, Kalman filter estimation accuracy and actuator commands.
# Five graphs and three CSV files are saved for each simulation run. RMSE is calculated to quantify tracking and estimation errors.

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from step5_Yara_Work.step4_pick_and_place import run_simulation

# ---- Simulation settings
# Select the cube position and whether to display MuJoCo.
# Same seed produces a reproducible cube starting position.
CUBE_SEED = 1
SHOW_VIEWER = True

# ---- Results directory
# Store evaluation results inside the Step 5 project folder.
RESULTS_ROOT = Path(__file__).resolve().parent / "evaluation_results"


# ---- Create results folder
# Create a separate folder for each simulation run.
def create_run_folder():

    # Create evaluation_results if it does not exist.
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    number = 1
    while True:
        folder = RESULTS_ROOT / f"run_{number:03d}"
        try:
            folder.mkdir()   # Create the next available run folder.
            return folder
        except FileExistsError:
            number += 1      # Increase run number if the folder already exists.


# ---- Save CSV files
# Store numerical measurements in spreadsheet-compatible files.
def save_csv(path, rows):

    # Skip saving if there are no measurements.
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        # Use dictionary keys as CSV column headings.
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---- Root-mean-square error
# RMSE = sqrt(mean(error^2)), calculated separately for each joint.
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


# ---- Plot one line per joint
# Shared layout for the Kalman error and actuator command graphs.
def plot_joints(times, values, ylabel, title, path):
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.plot(times, values[:, i], label=f"Joint {i + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)   # Save the graph as a PNG image.
    plt.close()


# ---- Generate performance graphs
# Evaluate PBVS, LQR and Kalman filter performance.
# Convert recorded simulation measurements into NumPy arrays.
def generate_graphs(data, folder):

    # Extract one column per joint (1-6) as a (samples x 6) array.
    def joints(prefix):
        return np.array([[row[f"{prefix}_{j}"] for j in range(1, 7)] for row in data])

    times = np.array([row["time_s"] for row in data])              # Simulation time [s].
    pbvs_errors = np.array([row["pbvs_error_m"] for row in data])  # PBVS Cartesian position error [m].
    q_desired = joints("q_desired")        # Desired joint positions [rad].
    q_actual = joints("q_actual")          # Actual joint positions from MuJoCo [rad].
    q_estimated = joints("q_estimated")    # Kalman-estimated joint positions [rad].
    dq_actual = joints("dq_actual")        # Actual joint velocities [rad/s].
    dq_estimated = joints("dq_estimated")  # Kalman-estimated joint velocities [rad/s].
    control = joints("control")            # LQR actuator position commands [rad].

    # ---- Graph 1: PBVS position error
    # Week 9: Visual servoing performance.
    # Error = ||target_position - end_effector_position||
    # Convert metres to millimetres for easier interpretation.
    plt.figure(figsize=(10, 5))
    plt.plot(times, pbvs_errors * 1000)
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (mm)")
    plt.title("PBVS End-Effector Position Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "pbvs_position_error.png", dpi=200)
    plt.close()

    # ---- Graph 2: LQR joint tracking
    # Week 6: State-feedback control.
    # Compare desired and actual positions for all six joints.
    # Smaller differences indicate better tracking.
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        ax.plot(times, q_desired[:, i], label="Desired")
        ax.plot(times, q_actual[:, i], label="Actual")
        ax.set_title(f"Joint {i + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True)
        ax.legend()
    fig.suptitle("LQR Joint Tracking")
    plt.tight_layout()
    plt.savefig(folder / "lqr_joint_tracking.png", dpi=200)
    plt.close()

    # ---- Graph 3: Kalman position error
    # Week 7: State estimation.
    # Position estimation error = q_estimated - q_actual
    # Error close to zero indicates accurate estimation.
    plot_joints(times, q_estimated - q_actual, "Estimation error (rad)",
                "Kalman Joint Position Estimation Error", folder / "kalman_position_error.png")

    # ---- Graph 4: Kalman velocity error
    # Week 7: State estimation.
    # Velocity estimation error = dq_estimated - dq_actual
    # Evaluates how accurately the filter estimates joint speeds.
    plot_joints(times, dq_estimated - dq_actual, "Velocity error (rad/s)",
                "Kalman Joint Velocity Estimation Error", folder / "kalman_velocity_error.png")

    # ---- Graph 5: Actuator commands
    # Week 6: LQR control output.
    # Display the position commands sent to each joint actuator.
    # Units are radians, not motor torque.
    plot_joints(times, control, "Position actuator command (rad)",
                "LQR Actuator Commands", folder / "control_effort.png")

    # ---- RMSE calculations
    # Week 6: LQR tracking error = q_desired - q_actual
    # Week 7: Kalman position error = q_estimated - q_actual
    # Week 7: Kalman velocity error = dq_estimated - dq_actual
    tracking_rmse = rmse(q_desired, q_actual)
    position_rmse = rmse(q_estimated, q_actual)
    velocity_rmse = rmse(dq_estimated, dq_actual)

    # ---- Save joint RMSE results
    # Save tracking and estimation RMSE for all six joints.
    rmse_rows = [{
        "Joint": i + 1,
        "Tracking_RMSE_rad": tracking_rmse[i],
        "Kalman_Position_RMSE_rad": position_rmse[i],
        "Kalman_Velocity_RMSE_rad_s": velocity_rmse[i],
    } for i in range(6)]
    save_csv(folder / "joint_rmse.csv", rmse_rows)

    # ---- Return mean RMSE results
    # Arithmetic mean of the six joint RMSE values.
    return {
        "mean_tracking_rmse_rad": float(np.mean(tracking_rmse)),
        "mean_kalman_position_rmse_rad": float(np.mean(position_rmse)),
        "mean_kalman_velocity_rmse_rad_s": float(np.mean(velocity_rmse)),
    }


# ---- Main evaluation function
# Run Step 4, collect measurements and generate performance results.
def main():

    # Create a new folder for this evaluation run.
    folder = create_run_folder()

    print("\n---- STEP 5 - PERFORMANCE EVALUATION")
    print("Results folder:", folder)

    # ---- Run pick-and-place simulation
    # Integrated control system.
    # Enable performance logging in Step 4.
    results = run_simulation(log_performance=True, cube_seed=CUBE_SEED, show_viewer=SHOW_VIEWER)

    # Retrieve recorded simulation measurements.
    data = results["performance_data"]
    if not data:
        raise RuntimeError("No performance measurements collected.")

    # ---- Save detailed measurements
    # Time, stage, PBVS error, joint states and actuator commands.
    save_csv(folder / "detailed_performance.csv", data)

    # ---- Generate graphs and RMSE
    # Evaluate visual servoing, LQR tracking and Kalman estimation.
    rmse_results = generate_graphs(data, folder)

    # ---- Performance summary
    # Combine task results with tracking and estimation accuracy.
    summary_keys = ["completed", "final_stage", "cube_seed", "task_duration_s",
                    "clipping_steps", "placement_error_m", "max_lqr_eigenvalue"]
    summary = {key: results[key] for key in summary_keys}
    summary.update(rmse_results)

    # Convert summary dictionary into CSV rows.
    save_csv(folder / "performance_summary.csv",
             [{"Metric": key, "Value": value} for key, value in summary.items()])

    # Display task success, placement error and RMSE values.
    print("\n---- STEP 5 RESULTS")
    print("Task completed:", results["completed"])
    print("Final stage:", results["final_stage"])
    print("Placement error:", round(results["placement_error_m"] * 1000, 3), "mm")    # Final cube placement error [mm].
    print("Actuator clipping steps:", results["clipping_steps"])                       # Timesteps with clipped commands.
    print("Mean joint tracking RMSE:", round(rmse_results["mean_tracking_rmse_rad"], 6), "rad")
    print("Mean Kalman position RMSE:", round(rmse_results["mean_kalman_position_rmse_rad"], 6), "rad")
    print("Mean Kalman velocity RMSE:", round(rmse_results["mean_kalman_velocity_rmse_rad_s"], 6), "rad/s")
    print("\nResults saved to:")
    print(folder)
    print("\nStep 5 evaluation complete.")

if __name__ == "__main__":
    main()