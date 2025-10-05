import tensorflow as tf
import matplotlib.pyplot as plt


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
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train, epochs=5, validation_data=(x_test, y_test)
)  # użyj verbose=0 jeśli jest problem z konsolą
model.evaluate(x_test, y_test)

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], color="blue", label="train")
plt.plot(history.history["val_loss"], color="green", label="validaton")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], color="blue", label="train")
plt.plot(history.history["val_accuracy"], color="green", label="validaton")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
