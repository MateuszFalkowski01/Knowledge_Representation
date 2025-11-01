import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import argparse


def rotate_around(x, y, angle_deg):
    theta = tf.constant(np.deg2rad(angle_deg), dtype=tf.float32)

    rotation_matrix = tf.stack(
        [
            [tf.math.cos(theta), -tf.math.sin(theta)],
            [tf.math.sin(theta), tf.math.cos(theta)],
        ]
    )

    point = tf.constant([[x], [y]], dtype=tf.float32)
    rotated = tf.matmul(rotation_matrix, point)
    result = tf.reshape(rotated, (2,))
    return result[0], result[1]


def main():
    parser = argparse.ArgumentParser(description="Rotate point (x, y) for a.")
    parser.add_argument("x", type=float, help="x")
    parser.add_argument("y", type=float, help="y")
    parser.add_argument("a", type=float, help="angle")

    args = parser.parse_args()
    x_rot, y_rot = rotate_around(args.x, args.y, args.a)
    print(f"Rotated point: ({x_rot:.4f}, {y_rot:.4f})")


if __name__ == "__main__":
    main()
