import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import argparse
import numpy as np


def rotate_point(x, y):
    # Konwersja na radiany
    theta = tf.constant(np.deg2rad(90), dtype=tf.float32)
    matrix2x2 = tf.stack(
        [
            [tf.math.cos(theta), -tf.math.sin(theta)],
            [tf.math.sin(theta), tf.math.cos(theta)],
        ]
    )

    # Zamiana na tensor kolumowy
    point = tf.constant([[x], [y]], dtype=tf.float32)
    rotated = tf.matmul(matrix2x2, point)
    return rotated[0, 0], rotated[1, 0]


def main():
    parser = argparse.ArgumentParser(description="Rotate point (x, y)")
    parser.add_argument("x", type=float, help="x")
    parser.add_argument("y", type=float, help="y")

    args = parser.parse_args()
    x_rot, y_rot = rotate_point(args.x, args.y)
    print(f"Rotated point: ({x_rot:.4f}, {y_rot:.4f})")


if __name__ == "__main__":
    main()
