import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from step5_Yara_Work.step4_pick_and_place import (
    run_simulation
)

CUBE_SEED = 1
SHOW_VIEWER = True

RESULTS_ROOT = (
    Path(__file__).resolve().parent
    / "evaluation_results"
)

def create_run_folder():

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    number = 1

    while True:

        folder = (
            RESULTS_ROOT /
            f"run_{number:03d}"
        )

        try:
            folder.mkdir()
            return folder

        except FileExistsError:
            number += 1


def save_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def generate_graphs(data, folder):

    times = np.array([
        row["time_s"] for row in data
    ])

    pbvs_errors = np.array([
        row["pbvs_error_m"] for row in data
    ])

    q_desired = np.array([
        [row[f"q_desired_{j}"] for j in range(1, 7)]
        for row in data
    ])

    q_actual = np.array([
        [row[f"q_actual_{j}"] for j in range(1, 7)]
        for row in data
    ])

    q_estimated = np.array([
        [row[f"q_estimated_{j}"] for j in range(1, 7)]
        for row in data
    ])

    dq_actual = np.array([
        [row[f"dq_actual_{j}"] for j in range(1, 7)]
        for row in data
    ])

    dq_estimated = np.array([
        [row[f"dq_estimated_{j}"] for j in range(1, 7)]
        for row in data
    ])

    control = np.array([
        [row[f"control_{j}"] for j in range(1, 7)]
        for row in data
    ])

    plt.figure(figsize=(10, 5))

    plt.plot(
        times,
        pbvs_errors * 1000
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position error (mm)")
    plt.title("PBVS End-Effector Position Error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        folder / "pbvs_position_error.png",
        dpi=200
    )

    plt.close()

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10)
    )

    for i, ax in enumerate(axes.flat):

        ax.plot(
            times,
            q_desired[:, i],
            label="Desired"
        )

        ax.plot(
            times,
            q_actual[:, i],
            label="Actual"
        )

        ax.set_title(f"Joint {i + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True)
        ax.legend()

    fig.suptitle("LQR Joint Tracking")

    plt.tight_layout()

    plt.savefig(
        folder / "lqr_joint_tracking.png",
        dpi=200
    )

    plt.close()


    plt.figure(figsize=(10, 5))

    for i in range(6):

        error = (
            q_estimated[:, i] -
            q_actual[:, i]
        )

        plt.plot(
            times,
            error,
            label=f"Joint {i + 1}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Estimation error (rad)")
    plt.title("Kalman Joint Position Estimation Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        folder / "kalman_position_error.png",
        dpi=200
    )

    plt.close()


    plt.figure(figsize=(10, 5))

    for i in range(6):

        error = (
            dq_estimated[:, i] -
            dq_actual[:, i]
        )

        plt.plot(
            times,
            error,
            label=f"Joint {i + 1}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity error (rad/s)")
    plt.title("Kalman Joint Velocity Estimation Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        folder / "kalman_velocity_error.png",
        dpi=200
    )

    plt.close()


    plt.figure(figsize=(10, 5))

    for i in range(6):

        plt.plot(
            times,
            control[:, i],
            label=f"Joint {i + 1}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Position actuator command (rad)")
    plt.title("LQR Actuator Commands")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        folder / "control_effort.png",
        dpi=200
    )

    plt.close()

    tracking_rmse = np.sqrt(
        np.mean(
            (q_desired - q_actual) ** 2,
            axis=0
        )
    )

    position_rmse = np.sqrt(
        np.mean(
            (q_estimated - q_actual) ** 2,
            axis=0
        )
    )

    velocity_rmse = np.sqrt(
        np.mean(
            (dq_estimated - dq_actual) ** 2,
            axis=0
        )
    )

    rmse_rows = []

    for i in range(6):

        rmse_rows.append({
            "Joint": i + 1,
            "Tracking_RMSE_rad": tracking_rmse[i],
            "Kalman_Position_RMSE_rad": position_rmse[i],
            "Kalman_Velocity_RMSE_rad_s": velocity_rmse[i]
        })

    save_csv(
        folder / "joint_rmse.csv",
        rmse_rows
    )

    return {
        "mean_tracking_rmse_rad": float(
            np.mean(tracking_rmse)
        ),
        "mean_kalman_position_rmse_rad": float(
            np.mean(position_rmse)
        ),
        "mean_kalman_velocity_rmse_rad_s": float(
            np.mean(velocity_rmse)
        )
    }

def main():

    folder = create_run_folder()

    print("\n================================")
    print("STEP 5 - PERFORMANCE EVALUATION")
    print("================================")

    print("Results folder:", folder)


    results = run_simulation(
        log_performance=True,
        cube_seed=CUBE_SEED,
        show_viewer=SHOW_VIEWER
    )

    data = results["performance_data"]

    if not data:
        raise RuntimeError(
            "No performance measurements collected."
        )


    save_csv(
        folder / "detailed_performance.csv",
        data
    )

    rmse = generate_graphs(
        data,
        folder
    )

    summary = {
        "completed": results["completed"],
        "final_stage": results["final_stage"],
        "cube_seed": results["cube_seed"],
        "task_duration_s": results["task_duration_s"],
        "clipping_steps": results["clipping_steps"],
        "placement_error_m": results["placement_error_m"],
        "max_lqr_eigenvalue": results["max_lqr_eigenvalue"],
        **rmse
    }

    summary_rows = [
        {"Metric": key, "Value": value}
        for key, value in summary.items()
    ]

    save_csv(
        folder / "performance_summary.csv",
        summary_rows
    )

    print("\n================================")
    print("STEP 5 RESULTS")
    print("================================")

    print("Task completed:", results["completed"])
    print("Final stage:", results["final_stage"])

    print(
        "Placement error:",
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
        "Mean joint tracking RMSE:",
        round(
            rmse["mean_tracking_rmse_rad"],
            6
        ),
        "rad"
    )

    print(
        "Mean Kalman position RMSE:",
        round(
            rmse["mean_kalman_position_rmse_rad"],
            6
        ),
        "rad"
    )

    print(
        "Mean Kalman velocity RMSE:",
        round(
            rmse["mean_kalman_velocity_rmse_rad_s"],
            6
        ),
        "rad/s"
    )

    print("\nResults saved to:")
    print(folder)

    print("\nStep 5 evaluation complete.")


if __name__ == "__main__":
    main()
