import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from PIL import Image

MODEL_PATH = "model.keras"


def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading path: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Training model.")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            x_train, y_train, epochs=5, validation_data=(x_test, y_test)
        )

        model.evaluate(x_test, y_test)
        model.save(MODEL_PATH)
        print(f"Model saved as {MODEL_PATH}.")

        plt.figure(figsize=(13, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], color="blue", label="train")
        plt.plot(history.history["val_loss"], color="green", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], color="blue", label="train")
        plt.plot(history.history["val_accuracy"], color="green", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    return model


def prepare_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img_arr = np.array(img) / 255.0

    # Wykrywanie odwrócenia kolorów dla czarnego tła
    mean_val = np.mean(img_arr)
    if mean_val < 0.5:
        img_arr = 1 - img_arr

    img_arr = img_arr.reshape(1, 28, 28)
    return img_arr


def predict_image(model, image_path):
    img_arr = prepare_image(image_path)
    prediction = model.predict(img_arr)
    digit = np.argmax(prediction)

    plt.imshow(img_arr.reshape(28, 28), cmap="gray")
    plt.title(f"Detected number: {digit}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train or use model.")
    parser.add_argument("--i", type=str, help="Path to image")
    args = parser.parse_args()

    model = load_model()

    if args.i:
        predict_image(model, args.i)
    else:
        print("Model ready. Use --i <Path> to use.")


if __name__ == "__main__":
    main()
