"""Generate AprilTag 36h11 marker images for the SO-ARM100 scene.

Creates 400x400 px PNGs (with a white border) for ids 0..3 using OpenCV's
aruco module and saves them to third_party/so_arm100/assets/apriltags/.
"""

import os

import cv2
import numpy as np

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "so_arm100",
    "assets",
    "apriltags",
)

IMAGE_SIZE = 400
WHITE_BORDER = 20
TAG_IDS = [0, 1, 2, 3]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    for tag_id in TAG_IDS:
        inner = IMAGE_SIZE - 2 * WHITE_BORDER
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, inner)
        image = cv2.copyMakeBorder(
            marker,
            WHITE_BORDER,
            WHITE_BORDER,
            WHITE_BORDER,
            WHITE_BORDER,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        image = np.ascontiguousarray(image, dtype=np.uint8)
        path = os.path.join(OUT_DIR, f"tag36h11_{tag_id}.png")
        cv2.imwrite(path, image)
        print(f"Wrote {path} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()
