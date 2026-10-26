"""
step5_calibration.py -- Task 1 (Aalia): calibrate the wrist camera from checkerboard images.

Zhang's method (Week 3): show a planar checkerboard of known square size at many
orientations, detect its internal corners, and solve for K (and lens distortion).
The arm holds the camera still and the board is moved in front of it, which is
equivalent to moving the camera around a fixed board.

The calibrated K is saved to results/calib.npz and is what the perception pipeline
uses. The "true" K from the MuJoCo camera's fovy is printed only for comparison.

Run (from the repo root):  python step3_Aalia_Work/step5_calibration.py
"""

import cv2
import numpy as np
import mujoco

import step1_Main.environment as env
from step3_Aalia_Work.step1_camera import WristCamera, noisy_rgb, pose_arm_to_look_down, RESULTS
from step3_Aalia_Work.step2_frames import T_world_cam, make_T

N_VIEWS = 15
PATTERN = (env.BOARD_SQUARES[0] - 1, env.BOARD_SQUARES[1] - 1)    # internal corners (across, down)
CALIB_FILE = RESULTS / "calib.npz"


def board_points():
    """Internal corners in the board frame (the z = 0 plane), in OpenCV's row-by-row order."""
    cols, rows = PATTERN
    i, j = np.meshgrid(np.arange(cols), np.arange(rows))
    P = np.stack([i - (cols - 1) / 2, j - (rows - 1) / 2, np.zeros_like(i)], axis=-1)
    return (P.reshape(-1, 3) * env.BOARD_SQUARE).astype(np.float32)


def find_corners(rgb):
    """Sub-pixel internal corners (N x 1 x 2), or None if the whole board is not found."""
    ok, corners = cv2.findChessboardCornersSB(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), PATTERN)
    return corners if ok else None


def random_board_pose(T_wc, rng):
    """Board pose in the world: 17-22 cm in front of the camera, facing it, tilted up to ~20 deg."""
    R_cb = np.diag([1.0, -1.0, -1.0]) @ cv2.Rodrigues(rng.uniform(-0.35, 0.35, 3))[0]   # board +z faces the camera
    t_cb = [*rng.uniform(-0.01, 0.01, 2), rng.uniform(0.17, 0.22)]
    return T_wc @ make_T(R_cb, t_cb)


def place_board(model, data, T_wb):
    m = model.body(env.BOARD_NAME).mocapid[0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T_wb[:3, :3].ravel())
    data.mocap_pos[m], data.mocap_quat[m] = T_wb[:3, 3], q
    mujoco.mj_forward(model, data)


def calibrate(corner_sets, image_size):
    """Returns (RMS reprojection error [px], K, distortion coefficients)."""
    obj = [board_points()] * len(corner_sets)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, corner_sets, image_size, None, None)
    return rms, K, dist


if __name__ == "__main__":
    model = env.build_scene(calibration_board=True)
    model.light_castshadow[:] = 0                     # otherwise the arm's shadow falls across the board
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    rng = np.random.default_rng(0)
    corner_sets, tries = [], 0
    RESULTS.mkdir(exist_ok=True)

    with WristCamera(model) as cam:
        pose_arm_to_look_down(model, data, cam.cam_id)
        T_wc = T_world_cam(data, cam.cam_id)
        for _ in range(4 * N_VIEWS):                      # some random poses will not show the whole board
            place_board(model, data, random_board_pose(T_wc, rng))
            rgb = noisy_rgb(cam.render_rgb(data), rng)
            if tries == 0:
                cv2.imwrite(str(RESULTS / "calibration_view.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            tries += 1
            corners = find_corners(rgb)
            if corners is not None:
                corner_sets.append(corners)
            if len(corner_sets) == N_VIEWS:
                break
        K_true = cam.K

    rms, K, dist = calibrate(corner_sets, (env.IMG_W, env.IMG_H))
    np.savez(CALIB_FILE, K=K, dist=dist, rms=rms)

    f = lambda K: np.round([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 2)
    print(f"board found in {len(corner_sets)} of {tries} views, RMS reprojection error {rms:.3f} px")
    print("fx, fy, cx, cy  calibrated:", f(K), "  true (from fovy):", f(K_true))
    print("distortion k1 k2 p1 p2 k3:", np.round(dist.ravel(), 4))