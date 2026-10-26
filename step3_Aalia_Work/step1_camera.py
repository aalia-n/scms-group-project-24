"""
camera.py -> the wrist-mounted RGB-D camera.

  WristCamera            renders RGB and depth (metres) and holds K
  project_points()       camera-frame 3D points -> pixels      (Week 3)
  backproject()          pixels + depth -> camera-frame points  (Week 4)
  pose_arm_to_look_down, table_point_on_optical_axis    viewing helpers for tests/evaluation

The camera itself is defined in 1_Main/environment.py (part of the shared scene).
Camera-frame convention here is OpenCV / Week 3: x right, y down, z forward.

camera.py   -> renders one view and saves results/camera_check.png
"""

from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
import step1_Main.environment as env 

RESULTS = HERE / "results"
MAX_DEPTH = 3.0                       # m; further than this counts as "no return" (depth 0)


# ---------------------------------------------------------------- pinhole model
def intrinsics_from_fovy(fovy_deg, width, height) -> np.ndarray:
    """f = (H/2) / tan(fovy/2); square pixels; principal point at the image centre (OpenCV pixel convention)."""
    f = (height / 2) / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2],
                     [0, f, (height - 1) / 2],
                     [0, 0, 1.0]])


def project_points(K, P_C):
    """3 x N camera-frame points -> (uv 2 x N, depth Z). Points behind the camera give nan."""
    P = np.asarray(P_C, dtype=float).reshape(3, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (K @ P)[:2] / P[2]
    uv[:, P[2] <= 0] = np.nan
    return uv, P[2]


def backproject(uv, depth, K) -> np.ndarray:
    """p_C = d K^-1 [u v 1]^T. 'depth' is planar depth (camera Z), which is what the renderer gives."""
    uv = np.asarray(uv, dtype=float).reshape(2, -1)
    uv1 = np.vstack([uv, np.ones(uv.shape[1])])
    return (np.linalg.inv(K) @ uv1) * np.asarray(depth, dtype=float).ravel()


# ---------------------------------------------------------------- rendering
class WristCamera:
    """
    with WristCamera(model) as cam:
        rgb, depth = cam.render_rgbd(data)     # uint8 H x W x 3, float H x W [m]
    """

    def __init__(self, model, name=env.CAM_NAME, width=env.IMG_W, height=env.IMG_H):
        self.name = name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.K = intrinsics_from_fovy(model.cam_fovy[self.cam_id], width, height)

    def render_rgb(self, data):
        self.renderer.update_scene(data, camera=self.name)
        return self.renderer.render().copy()

    def render_depth(self, data):
        """Planar depth in metres; 0 where there is no valid return."""
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().astype(float)
        self.renderer.disable_depth_rendering()
        depth[~np.isfinite(depth) | (depth >= MAX_DEPTH)] = 0.0
        return depth

    def render_rgbd(self, data):
        return self.render_rgb(data), self.render_depth(data)

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- viewing helpers
def optical_axis_world(data, cam_id):
    """Unit viewing direction in the world frame (MuJoCo cameras look along -z)."""
    return -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]


def pose_arm_to_look_down(model, data, cam_id):
    """
    Set the two wrist joints so the camera looks as close to straight down as possible.
    Sets qpos and ctrl, so the position actuators hold the pose when the simulation steps.
    """
    a1 = model.joint("wrist_1_joint").qposadr[0]
    a2 = model.joint("wrist_2_joint").qposadr[0]
    best, best_q = np.inf, (data.qpos[a1], data.qpos[a2])
    for q1 in np.linspace(-np.pi, np.pi, 17):
        for q2 in np.linspace(-np.pi, np.pi, 17):
            data.qpos[a1], data.qpos[a2] = q1, q2
            mujoco.mj_forward(model, data)
            down = optical_axis_world(data, cam_id)[2]
            if data.cam_xpos[cam_id][2] - env.TABLE_TOP_Z > 0.15 and down < best:
                best, best_q = down, (q1, q2)
    data.qpos[a1], data.qpos[a2] = best_q
    data.ctrl[model.actuator("wrist_1").id], data.ctrl[model.actuator("wrist_2").id] = best_q
    mujoco.mj_forward(model, data)


def table_point_on_optical_axis(data, cam_id, z=env.TABLE_TOP_Z + env.OBJ_HALF):
    """Where the optical axis meets the horizontal plane at height z (world frame)."""
    c, a = data.cam_xpos[cam_id], optical_axis_world(data, cam_id)
    return c + (z - c[2]) / a[2] * a


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    with WristCamera(model) as cam:
        pose_arm_to_look_down(model, data, cam.cam_id)
        env.place_object(model, data, table_point_on_optical_axis(data, cam.cam_id)[:2])
        rgb, depth = cam.render_rgbd(data)
        print("K =\n", np.round(cam.K, 2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    im = ax[1].imshow(np.where(depth > 0, depth, np.nan)); ax[1].set_title("depth [m]")
    plt.colorbar(im, ax=ax[1], shrink=0.85)
    RESULTS.mkdir(exist_ok=True)
    plt.savefig(RESULTS / "camera_check.png", dpi=120)
    plt.show()