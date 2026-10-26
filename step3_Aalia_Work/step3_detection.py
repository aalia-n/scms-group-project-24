"""
detection.py ->  find the target object in the RGB image.

  detect_object(rgb) -> (mask, (u, v)) or None

Colour segmentation: RGB -> HSV, threshold red, clean up with morphology,
keep the largest blob. Returns None when nothing is found (object out of view,
too small or hidden), so callers must handle a missed detection.

Run:  python detection.py   -> detects the cube in one view and saves results/detection_check.png
"""

import cv2
import numpy as np

# Red wraps around hue 0 in OpenCV's 0-180 hue scale, so it needs two ranges.
RED_RANGES = [((0, 120, 50), (10, 255, 255)),
              ((170, 120, 50), (180, 255, 255))]
MIN_AREA = 50                      # px; smaller blobs are treated as noise
KERNEL = np.ones((5, 5), np.uint8)


def detect_object(rgb):
    """RGB uint8 image -> (boolean mask of the object, centroid (u, v)), or None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)    # remove speckles
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)   # fill small holes

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:                                                # label 0 is the background
        return None
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
        return None
    return labels == i, centroids[i]


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mujoco
    import step3_Aalia_Work.step1_camera as cm
    import step1_Main.environment as env

    model = env.build_scene()
    data = mujoco.MjData(model)
    env.reset_home(model, data)
    with cm.WristCamera(model) as cam:
        cm.pose_arm_to_look_down(model, data, cam.cam_id)
        env.place_object(model, data, cm.table_point_on_optical_axis(data, cam.cam_id)[:2] + [0.04, 0.03])
        rgb = cam.render_rgb(data)
        env.place_object(model, data, (5.0, 5.0))             # out of view
        missing = detect_object(cam.render_rgb(data))

    mask, (u, v) = detect_object(rgb)
    print(f"centroid ({u:.1f}, {v:.1f}) px, {mask.sum()} px;  object out of view -> {missing}")

    plt.imshow(rgb)
    plt.contour(mask, colors="y")
    plt.plot(u, v, "c+", ms=14, mew=2)
    cm.RESULTS.mkdir(exist_ok=True)
    plt.savefig(cm.RESULTS / "detection_check.png", dpi=120)
    plt.show()