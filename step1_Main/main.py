'''
main.py ---> whole integrated pipeline, shared by all team members 
'''

import time
import numpy as np
import mujoco
import mujoco.viewer

import step1_Main.environment as env                                    # noqa: E402
import step3_Aalia_Work.step1_camera as camera
from step3_Aalia_Work.step3_detection import detect_object
from step3_Aalia_Work.step4_localisation import localise
from step3_Aalia_Work.step2_frames import T_world_cam
# from step4_Sameeha_Work.observer import JointObserver      # Sameeha
# from step5_Yara_Work.controller import PickController      # Yara

PERCEPTION_PERIOD = 0.1     # s; camera runs at 10 Hz, much slower than the physics


def main(seed=None):
    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    env.randomise_object(model, data, np.random.default_rng(seed))

    cam = camera.WristCamera(model)
    camera.pose_arm_to_look_down(model, data, cam.cam_id)   # temporary: until Yara's approach motion exists
    # observer = JointObserver(model)                        # Sameeha
    # controller = PickController(model)                     # Yara

    target = None              # latest perception result: (p_W, p_C) or None
    next_perception, next_print = 0.0, 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # ---- perception (Aalia): 10 Hz
            if data.time >= next_perception:
                next_perception += PERCEPTION_PERIOD
                rgb, depth = cam.render_rgbd(data)
                det = detect_object(rgb)
                target = None if det is None else localise(det[0], depth, cam.K, T_world_cam(data, cam.cam_id))

            # ---- estimation (Sameeha)
            # x_hat, P = observer.update(data.sensordata, data.ctrl)

            # ---- control (Yara)
            # data.ctrl[:] = controller.step(target, x_hat)

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_print:
                next_print += 1.0
                print(f"t = {data.time:5.1f} s   target (world) = "
                      f"{'not detected' if target is None else np.round(target[0], 3)}")

            # real-time pacing
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    cam.close()


if __name__ == "__main__":
    main()