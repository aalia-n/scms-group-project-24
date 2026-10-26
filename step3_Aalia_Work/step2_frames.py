"""
frames.py -- Task 1 (Aalia): frame management, camera frame <-> world frame.

Frame chain:
    {W} world -> {B} base -> {E} wrist_3_link (forward kinematics, q) -> {Cmj} MuJoCo camera
    -> {C} OpenCV camera  (MJ_TO_CV: MuJoCo has y up / looks along -z, OpenCV has y down / looks along +z)

    p_W = W_T_C p_C

W_T_C comes from the joint angles (MuJoCo's forward kinematics), so it is a
measurement, not ground truth. Transforms are 4x4 arrays; point sets are 3 x N.

Run:  python frames.py   -> localises the cube from one RGB-D frame and compares with ground truth
"""

import numpy as np

MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def invert_T(T):
    """(A_T_B)^-1 = [R^T, -R^T t]."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, P):
    """Apply T to a point (3,) or points (3 x N); returns the same shape."""
    P = np.asarray(P, dtype=float)
    out = T[:3, :3] @ P.reshape(3, -1) + T[:3, 3:4]
    return out.ravel() if P.ndim == 1 else out


def T_world_cam(data, cam_id):
    """W_T_C in the OpenCV convention (matches K). Needs mj_forward after qpos changes."""
    R = data.cam_xmat[cam_id].reshape(3, 3) @ MJ_TO_CV
    return make_T(R, data.cam_xpos[cam_id])


def cam_to_world(T_wc, P_C):
    return transform_points(T_wc, P_C)


def world_to_cam(T_wc, P_W):
    return transform_points(invert_T(T_wc), P_W)


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import mujoco
    import step3_Aalia_Work.step1_camera as cm
    import step1_Main.environment as env          

    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)

    with cm.WristCamera(model) as cam:
        cm.pose_arm_to_look_down(model, data, cam.cam_id)
        T_wc = T_world_cam(data, cam.cam_id)
        # offset the cube from the optical axis so the x/y directions are exercised
        hit = cm.table_point_on_optical_axis(data, cam.cam_id)
        env.place_object(model, data, hit[:2] + 0.05 * T_wc[:2, 0] + 0.03 * T_wc[:2, 1])
        rgb, depth = cam.render_rgbd(data)

        # red pixels + depth -> camera frame -> world frame
        r, g, b = (rgb[..., i].astype(int) for i in range(3))
        v, u = np.nonzero((r > 60) & (r > 1.8 * g) & (r > 1.8 * b) & (depth > 0))
        P_W = cam_to_world(T_wc, cm.backproject(np.vstack([u, v]), depth[v, u], cam.K))
        p_est = np.median(P_W, axis=1)

    p_true = data.body(env.OBJ_NAME).xpos      # ground truth: evaluation only
    print("estimate  ", np.round(p_est, 4))
    print("true      ", np.round(p_true, 4))
    print("error [mm]", np.round(1000 * (p_est - p_true), 1),
          f"(z is about +{1000 * env.OBJ_HALF:.0f} mm: only the cube's top face is visible)")