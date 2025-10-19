import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import argparse
import ast


def solve_equal(A, b):
    A_tf = tf.constant(np.array(A, dtype=np.float64))
    b_tf = tf.constant(np.array(b, dtype=np.float64))

    # Dopilnowanie, że b jest kolumną (n×1)
    if len(b_tf.shape) == 1:
        b_tf = tf.reshape(b_tf, (-1, 1))

    # A * x = b
    x_tf = tf.linalg.solve(A_tf, b_tf)
    return x_tf.numpy().flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Solve a system of linear equations A * x = b using TensorFlow."
    )

    parser.add_argument(
        "--A",
        type=str,
        required=True,
        help='Matrix A in Python list format "[[3,1,-1]]"',
    )

    parser.add_argument(
        "--b", type=str, required=True, help='Vector b in Python list format "[4,1,1]"'
    )

    args = parser.parse_args()

    try:
        A = ast.literal_eval(args.A)
        b = ast.literal_eval(args.b)
    except Exception as e:
        print("Error while parsing input:", e)
        return

    x = solve_equal(A, b)
    print("Solution:", x)


if __name__ == "__main__":
    main()
