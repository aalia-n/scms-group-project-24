### environment.py

"""
  build_scene()        UR3e (from ur3e.xml) + wrist RGB-D camera + table + target object + light
  reset_home()         robot to keyframe 0 ("home")
  place_object()       put the object at (x, y) on the table, with a yaw angle
  randomise_object()   random (x, y, yaw) inside OBJ_REGION (the brief requires randomised trials)

Every team member builds the scene with this file, so everyone simulates the same world.
ur3e.xml itself is not modified: the extra items are added in Python with MjSpec.

environment.py   -> opens the viewer on the scene with a randomised object
"""

from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
ROBOT_XML = REPO / "step2_mujoco_UR3_tutorial_from_rafael" / "ur3e.xml"

# ---------------------------------------------------------------- wrist camera
CAM_NAME = "wrist_cam"
IMG_W, IMG_H = 640, 480
FOVY_DEG = 58.0                         # vertical FOV of a RealSense D435-class depth camera
WRIST_BODY = "wrist_3_link"
FLANGE_SITE = "attachment_site"         # tool flange; its +z points out of the tool
CAM_OFFSET_FLANGE = (0.0, 0.07, 0.02)   # camera position in the flange frame: 7 cm to the side, 2 cm forward

# ---------------------------------------------------------------- table and object
TABLE_TOP_Z = 0.001
TABLE_HALF = (0.6, 0.6, 0.02)
OBJ_NAME = "target"
OBJ_HALF = 0.02                         # 4 cm cube
OBJ_RGBA = (0.85, 0.10, 0.10, 1.0)
# Where the object may be placed: (x range, y range) [m]. Currently the area the
# camera sees from the look-down test pose; widen once the approach motion exists.
OBJ_REGION = ((0.05, 0.21), (-0.30, -0.15))


def build_scene(robot_xml=ROBOT_XML) -> mujoco.MjModel:
    base = mujoco.MjModel.from_xml_path(str(robot_xml))

    # Camera pose in the wrist frame: the flange frame turned 180 deg about x,
    # because MuJoCo cameras look along their -z and the flange's +z points out of the tool.
    site = base.site(FLANGE_SITE)
    R_site = np.zeros(9)
    mujoco.mju_quat2Mat(R_site, site.quat)
    R_site = R_site.reshape(3, 3)
    cam_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_quat, (R_site @ np.diag([1.0, -1.0, -1.0])).ravel())

    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, IMG_W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, IMG_H)

    cam = spec.body(WRIST_BODY).add_camera()
    cam.name, cam.fovy = CAM_NAME, FOVY_DEG
    cam.pos = site.pos + R_site @ np.array(CAM_OFFSET_FLANGE)
    cam.quat = cam_quat

    world = spec.worldbody
    table = world.add_geom()
    table.name, table.type, table.size = "table", mujoco.mjtGeom.mjGEOM_BOX, TABLE_HALF
    table.pos = (0.0, 0.0, TABLE_TOP_Z - TABLE_HALF[2])
    table.rgba = (0.62, 0.50, 0.36, 1.0)

    obj = world.add_body()
    obj.name, obj.pos = OBJ_NAME, (0.13, -0.22, TABLE_TOP_Z + OBJ_HALF)
    joint = obj.add_joint()                                   # free joint: can be moved and picked up
    joint.name, joint.type = OBJ_NAME + "_free", mujoco.mjtJoint.mjJNT_FREE
    box = obj.add_geom()
    box.name, box.type = OBJ_NAME + "_geom", mujoco.mjtGeom.mjGEOM_BOX
    box.size, box.rgba = (OBJ_HALF,) * 3, OBJ_RGBA

    light = world.add_light()
    light.pos, light.dir, light.diffuse = (0, 0, 2.0), (0, 0, -1.0), (0.7, 0.7, 0.7)

    # The free joint adds 7 qpos entries; extend the 6-entry keyframes in ur3e.xml to match.
    for key in spec.keys:
        if len(key.qpos) == base.nq:
            key.qpos = np.concatenate([key.qpos, [*obj.pos, 1, 0, 0, 0]])

    return spec.compile()


def reset_home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def place_object(model, data, xy, yaw=0.0):
    """Object upright at (x, y) on the table, rotated by yaw [rad] about the vertical."""
    a = model.joint(OBJ_NAME + "_free").qposadr[0]
    data.qpos[a:a + 7] = [xy[0], xy[1], TABLE_TOP_Z + OBJ_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d = model.joint(OBJ_NAME + "_free").dofadr[0]
    data.qvel[d:d + 6] = 0
    mujoco.mj_forward(model, data)


def randomise_object(model, data, rng):
    """Random position in OBJ_REGION and random yaw. Pass a seeded rng for reproducible trials."""
    (x0, x1), (y0, y1) = OBJ_REGION
    place_object(model, data, (rng.uniform(x0, x1), rng.uniform(y0, y1)), rng.uniform(-np.pi, np.pi))


if __name__ == "__main__":
    import mujoco.viewer

    model = build_scene()
    data = mujoco.MjData(model)
    reset_home(model, data)
    randomise_object(model, data, np.random.default_rng())
    mujoco.viewer.launch(model, data)
