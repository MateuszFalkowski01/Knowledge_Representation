import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import tensorflow as tf
import numpy as np


@tf.function
def solve_equal(A, b):
    # przekształcenie jednowymiarowego b
    if len(b.shape) == 1:
        b = tf.reshape(b, (-1, 1))
    x = tf.linalg.solve(A, b)
    return tf.reshape(x, [-1])


def parse_args(args):
    if len(args) < 2:
        raise ValueError(
            "Usage: python solve_system.py <a11> <a12> ... <ann> <b1> <b2> ... <bn>\n"
            "Example: python solve_system.py 3 1 -1 2 4 1 -1 2 5 4 1 1"
        )

    try:
        data = list(map(float, args))
    except ValueError:
        raise ValueError("All arguments have to be numbers.")

    length = len(data)
    n = None
    for i in range(1, 100):
        if i * i + i == length:
            n = i
            break

    if n is None:
        raise ValueError(
            f"The number of arguments ({length}) does not match any n (because n^2 + n != {length})."
        )

    A_vals = data[: n * n]
    b_vals = data[n * n :]
    A = np.array(A_vals).reshape((n, n))
    b = np.array(b_vals)

    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        raise ValueError("Irreversible matrix")

    return A, b


def run_solver():
    try:
        A, b = parse_args(sys.argv[1:])
        x = solve_equal(A, b)
        print("Result:")
        for i, val in enumerate(x, start=1):
            print(f"x{i} = {val:.6f}")
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)


def main():
    run_solver()


if __name__ == "__main__":
    main()
