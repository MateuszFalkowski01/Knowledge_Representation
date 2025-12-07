import tensorflow as tf
import keras
import numpy as np
import os
import json
import argparse
from sklearn.metrics import confusion_matrix
from keras import layers
from PIL import Image


MODEL_FILENAME = "model.keras"
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
EPOCHS = 10

# Train images dimensions: (60000, 28, 28)
# Test images dimensions: (10000, 28, 28)


def load_data(version):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if version == 1:
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

    return (train_images, train_labels), (test_images, test_labels)


data_augmentation = keras.Sequential(
    [
        # jawne określenie kształtu wejścia, żeby zapobiec błędom w modelu konwolucyjnym
        keras.Input(shape=(28, 28, 1)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)


def create_model_fully_connected():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            # dodanie wymiar kanału (28, 28) -> (28, 28, 1) żeby zapobiec błędom wczytywania modelu
            keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
            data_augmentation,
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createmodel_Convolutional():
    model = keras.Sequential(
        [
            data_augmentation,
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(filename, version):
    (train_images, train_labels), (test_images, test_labels) = load_data(version)

    if version == 0:
        model = create_model_fully_connected()
    else:
        model = createmodel_Convolutional()
    model.fit(train_images, train_labels, EPOCHS)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100}%")
    print(f"Model Loss: {test_loss}")

    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    cm = confusion_matrix(test_labels, predicted_classes)

    metrics_data = {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
    }

    model.save(filename)

    metrics_filename = filename.replace(".keras", ".json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics saved as: {metrics_filename}")

    return model


def prepare_models():
    connect = "connected_" + MODEL_FILENAME
    convo = "convolutional_" + MODEL_FILENAME

    if os.path.exists(connect):
        print(f"\nLoading saved model: '{connect}'.")
        model1 = keras.models.load_model(connect)
        load_metrics(connect)
    else:
        model1 = train_model(connect, 0)

    if os.path.exists(convo):
        print(f"\nLoading saved model: '{convo}'.")
        model2 = keras.models.load_model(convo)
        load_metrics(convo)
    else:
        model2 = train_model(convo, 1)

    return model1, model2


def load_metrics(filename):
    metrics_filename = filename.replace(".keras", ".json")
    if os.path.exists(metrics_filename):
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)

        cm = np.array(metrics["confusion_matrix"])

        print(f"\nLoading metrics from '{metrics_filename}'.")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print("Confusion Matrix:\n", cm)
        return metrics
    else:
        print(f"\nFile not found: '{metrics_filename}'.")
        return None


def process_and_predict_image(model, image_path):
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Wrong path: {image_path}")
        return
    except Exception as e:
        print(f"Error while loading file: {e}")
        return

    img = img.resize((28, 28))
    img_gray = img.convert("L")

    # normalizacja (0-255 -> 0.0-1.0)
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    # negatyw
    img_processed = 1.0 - img_array

    # dodawanie wymiarów batcha i kanału gdzie są potrzebne
    if len(img_processed.shape) == 2:
        input_tensor = np.expand_dims(img_processed, axis=0)

    if not isinstance(model.layers[0], keras.layers.Reshape):
        input_tensor = np.expand_dims(input_tensor, axis=-1)

    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    print(f"\nPrediction for image: '{image_path}'")
    print(f"Class: {predicted_class}")
    print(f"Probability: {confidence}%")


def main():
    parser = argparse.ArgumentParser(description="Image prediction.")
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to the image file for prediction.",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help="Choose model for prediction. Options: 0 - Fully Connected, 1 - CNN. Deafult: 0.",
    )

    args = parser.parse_args()

    model_fully_connected, model_convolutional = prepare_models()

    if args.predict:
        if args.model == 0:
            model = model_fully_connected
        else:
            model = model_convolutional

    process_and_predict_image(model, args.predict)


if __name__ == "__main__":
    main()
