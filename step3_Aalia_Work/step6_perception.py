"""
step6_perception.py -- Task 1 (Aalia): the perception interface the rest of the team uses.

  perception = ObjectPerception(model)
  m = perception.measure(data)        # Measurement, or None if the object is not seen
  target = TargetFilter()
  p, P = target.update(m)             # fused target estimate and its covariance

measure() runs the whole sensing chain on one frame:
  render RGB-D -> add sensor noise -> detect (step3) -> localise (step4) -> world frame (step2)
and attaches a measurement covariance from the noise model. It uses the CALIBRATED K
from results/calib.npz (step5); if that file does not exist yet, it falls back to the
true K and says so.

TargetFilter is a Kalman filter for a static target: repeated noisy measurements of an
object that does not move are fused into one estimate whose uncertainty shrinks.

Run (from the repo root):  python step3_Aalia_Work/step6_perception.py
"""

from dataclasses import dataclass

import numpy as np

from step3_Aalia_Work.step1_camera import WristCamera, noisy_rgb, noisy_depth, depth_sigma
from step3_Aalia_Work.step2_frames import T_world_cam
from step3_Aalia_Work.step3_detection import detect_object
from step3_Aalia_Work.step4_localisation import localise
from step3_Aalia_Work.step5_calibration import CALIB_FILE

SIGMA_PX = 1.0          # px; assumed uncertainty of the object's image position (sideways error)
Q_TARGET = 1e-8         # m^2 per update; tiny process noise: the target is assumed static


@dataclass
class Measurement:
    p_world: np.ndarray     # object centre in the world frame [m]       (for PBVS / approach planning)
    p_cam: np.ndarray       # object centre in the camera frame [m]      (p_cam[2] is its depth)
    uv: np.ndarray          # object centroid in the image [px]          (for IBVS)
    cov_world: np.ndarray   # 3x3 measurement covariance, world frame [m^2]
    t: float                # simulation time [s]


class ObjectPerception:
    def __init__(self, model, noise=True, seed=None):
        self.cam = WristCamera(model)
        if CALIB_FILE.exists():
            self.K = np.load(CALIB_FILE)["K"]
        else:
            print("perception: no calib.npz yet, using the true K (run step5_calibration.py)")
            self.K = self.cam.K
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def measure(self, data):
        rgb, depth = self.cam.render_rgbd(data)
        if self.noise:
            rgb, depth = noisy_rgb(rgb, self.rng), noisy_depth(depth, self.rng)

        det = detect_object(rgb)
        if det is None:
            return None
        mask, uv = det
        T_wc = T_world_cam(data, self.cam.cam_id)
        loc = localise(mask, depth, self.K, T_wc)
        if loc is None:
            return None
        p_world, p_cam = loc

        # Covariance in the camera frame: sideways from SIGMA_PX at this depth, along the
        # optical axis from the depth noise model; then rotated into the world frame.
        s_xy = p_cam[2] * SIGMA_PX / self.K[0, 0]
        s_z = depth_sigma(p_cam[2])
        R = T_wc[:3, :3]
        cov_world = R @ np.diag([s_xy ** 2, s_xy ** 2, s_z ** 2]) @ R.T
        return Measurement(p_world, p_cam, np.asarray(uv), cov_world, data.time)

    def close(self):
        self.cam.close()


class TargetFilter:
    """Kalman filter for a static 3D target in the world frame (state = position, model x_k+1 = x_k)."""

    def __init__(self):
        self.x, self.P = None, None

    def update(self, m):
        """Fuse one Measurement (or None, which keeps the current estimate). Returns (x, P)."""
        if m is None:
            return self.x, self.P
        if self.x is None:                                    # first sighting initialises the filter
            self.x, self.P = m.p_world.copy(), m.cov_world.copy()
            return self.x, self.P
        self.P = self.P + Q_TARGET * np.eye(3)                # predict: target has not moved
        G = self.P @ np.linalg.inv(self.P + m.cov_world)      # Kalman gain
        self.x = self.x + G @ (m.p_world - self.x)            # update
        self.P = (np.eye(3) - G) @ self.P
        return self.x, self.P


if __name__ == "__main__":
    import mujoco
    import step1_Main.environment as env
    from step3_Aalia_Work.step1_camera import pose_arm_to_look_down

    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    perception, target = ObjectPerception(model, seed=0), TargetFilter()
    pose_arm_to_look_down(model, data, perception.cam.cam_id)
    env.randomise_object(model, data, np.random.default_rng(1))

    for _ in range(20):
        m = perception.measure(data)
        x, P = target.update(m)
    perception.close()

    p_true = data.body(env.OBJ_NAME).xpos                     # ground truth: evaluation only
    mm = lambda v: np.round(1000 * v, 2)
    print("last measurement  error [mm]", mm(m.p_world - p_true), "  1-sigma [mm]", mm(np.sqrt(np.diag(m.cov_world))))
    print("filtered (20)     error [mm]", mm(x - p_true), "  1-sigma [mm]", mm(np.sqrt(np.diag(P))))