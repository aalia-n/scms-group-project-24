"""
localisation.py -> object mask + depth -> 3D position in the world frame.

  localise(mask, depth, K, T_wc) -> (p_W, p_C) or None

1. Back-project every object pixel with valid depth (p_C = d K^-1 [u v 1]^T)
   and take the median: robust to edge pixels and dropouts, unlike a single pixel.
2. Visible-surface correction: the camera only sees the near surface, so add half the
   object's depth to the estimate's depth (camera Z) to reach its centre. (Along camera Z,
   not along the pixel's ray: the ray is slanted for off-centre objects and would add a
   sideways error.)
3. Transform to the world frame with W_T_C from frames.py.

Run:  python localisation.py   -> 20 random cube positions, RMS error with and without the correction
"""

import numpy as np

from step1_camera import backproject, env
from step2_frames import cam_to_world


def localise(mask, depth, K, T_wc, half_depth=env.OBJ_HALF):
    """Returns (p_W, p_C), the object centre in the world and camera frames, or None."""
    v, u = np.nonzero(mask & (depth > 0))
    if u.size == 0:
        return None
    p_C = np.median(backproject(np.vstack([u, v]), depth[v, u], K), axis=1)
    p_C[2] += half_depth
    return cam_to_world(T_wc, p_C), p_C


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import mujoco
    import step1_camera as cm
    from step3__detection import detect_object
    from step2_frames import T_world_cam

    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    rng = np.random.default_rng(0)
    err = {"corrected": [], "uncorrected": []}

    with cm.WristCamera(model) as cam:
        cm.pose_arm_to_look_down(model, data, cam.cam_id)
        T_wc = T_world_cam(data, cam.cam_id)
        centre = cm.table_point_on_optical_axis(data, cam.cam_id)[:2]
        for _ in range(20):
            env.place_object(model, data, centre + rng.uniform(-0.08, 0.08, 2))
            rgb, depth = cam.render_rgbd(data)
            mask, _ = detect_object(rgb)
            p_true = data.body(env.OBJ_NAME).xpos         # ground truth: evaluation only
            err["corrected"].append(localise(mask, depth, cam.K, T_wc)[0] - p_true)
            err["uncorrected"].append(localise(mask, depth, cam.K, T_wc, half_depth=0)[0] - p_true)

    for name, e in err.items():
        e = 1000 * np.array(e)
        print(f"{name:12s} RMS error x, y, z [mm]: {np.round(np.sqrt((e ** 2).mean(axis=0)), 1)}")