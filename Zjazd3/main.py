import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import argparse
import sys

EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 0.001

raw_data = []
try:
    with open('wine.data', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            raw_data.append([float(x) for x in row])
except FileNotFoundError:
    print("Error: File not found 'wine.data'.")
    sys.exit(1)

data_np = np.array(raw_data)
y_raw = data_np[:, 0]
X_raw = data_np[:, 1:]

y_shifted = y_raw - 1
num_classes = 3
y_one_hot = np.eye(num_classes)[y_shifted.astype(int)]

indices = np.arange(X_raw.shape[0])
np.random.shuffle(indices)
X_shuffled = X_raw[indices]
y_shuffled = y_one_hot[indices]

split_idx = int(0.7 * len(X_shuffled))
X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]


model_simple = models.Sequential(name="Simple_Model")
model_simple.add(layers.Input(shape=(13,)))
model_simple.add(layers.Dense(64, activation='relu'))
model_simple.add(layers.Dropout(0.05))
model_simple.add(layers.Dense(32, activation='relu'))
model_simple.add(layers.Dense(3, activation='softmax'))
model_simple.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

model_complex = models.Sequential(name="Expanded_Model")
model_complex.add(layers.Input(shape=(13,)))
model_complex.add(layers.Dense(128, activation='elu'))
model_complex.add(layers.Dropout(0.3))
model_complex.add(layers.Dense(64, activation='elu'))
model_complex.add(layers.Dense(3, activation='softmax'))
model_complex.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

history_simple = model_simple.fit(X_train, y_train,
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  validation_data=(X_test, y_test),
                                  verbose=2)
history_complex = model_complex.fit(X_train, y_train,
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=(X_test, y_test),
                                    verbose=2)

def learning_plot(hist, title):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.title(f'{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.title(f'{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

parser = argparse.ArgumentParser(description='Wine Classification.')
parser.add_argument('--predict', 
                    type=float, 
                    nargs=13, 
                    metavar='Attribute',
                    help='Provide 13 attributes for wine prediction.')

args, unknown = parser.parse_known_args() 

loss1, acc1 = model_simple.evaluate(X_test, y_test, verbose=0)
loss2, acc2 = model_complex.evaluate(X_test, y_test, verbose=0)

print(f"\nSimple Model:  Loss: {loss1:.4f}, Accuracy: {acc1:.4f}")
print(f"Complex Model: Loss: {loss2:.4f}, Accuracy: {acc2:.4f}")

learning_plot(history_simple, "Simple Model")
learning_plot(history_complex, "Complex Model")

if args.predict:
    input = np.array(args.predict)
    input_reshaped = input.reshape(1, -1)
    prediction_prob = model_complex.predict(input_reshaped, verbose=0)
    predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
    predicted_class_label = predicted_class_index + 1
    print(f"Predicted class: {predicted_class_label}")
