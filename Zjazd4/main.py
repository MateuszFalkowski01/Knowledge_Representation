import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import keras_tuner as kt
import argparse
import sys
import os 

model_filename = 'model.keras'
SKIP_TRAINING = False


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

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train)


def create_model(units, dropout_rate, learning_rate):

    model = models.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(normalizer)
    model.add(layers.Dense(units, activation='relu'))    
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_hypermodel(hp):
    """
    Funkcja obsługi hiperparametów z Keras Tuner.
    Poszukiwane hiperparametry:
    1. units: liczba neuronów w pierwszej warstwie
    2. dropout_rate: współczynnik zapominania
    3. learning_rate: tempo uczenia
    """
    
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    return create_model(units=hp_units, 
                        dropout_rate=hp_dropout, 
                        learning_rate=hp_lr)


if os.path.exists(model_filename):
    print(f"\nLoading saved model: '{model_filename}'.")
    best_model = tf.keras.models.load_model(model_filename)
    history = tf.keras.callbacks.History() 
    SKIP_TRAINING = True
else:
    
    tuner = kt.Hyperband(
        build_hypermodel,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='kt_dir',
        project_name='wine_tuning',
    )
    
    tuner.search(X_train, y_train, 
                 epochs=50, 
                 validation_data=(X_test, y_test),
                 verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Ostateczne trenowanie z najlepszymi parametrami
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train, y_train, 
                             epochs=80, 
                             validation_data=(X_test, y_test),
                             verbose=2)

    best_model.save(model_filename)
    print(f"\nModel saved as: {model_filename}")


if not SKIP_TRAINING:
    print(f"""
Found optimal hyperparameters:
- Units: {best_hps.get('units')}
- Dropout: {best_hps.get('dropout')}
- Learning Rate: {best_hps.get('learning_rate')}
""")


def learning_plot(hist):
    if not hist.history: 
        return
        
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


loss_eval, acc_eval = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBest Model Evaluation: Loss: {loss_eval:.4f}, Accuracy: {acc_eval:.4f}")
learning_plot(history)

parser = argparse.ArgumentParser(description='Wine Classification.')
parser.add_argument('--predict', 
                    type=float, 
                    nargs=13, 
                    metavar='Attribute',
                    help='Provide 13 attributes for wine prediction.')

args, unknown = parser.parse_known_args() 

if args.predict:
    
    input_data = np.array(args.predict)
    input_reshaped = input_data.reshape(1, -1)
    
    prediction_prob = best_model.predict(input_reshaped, verbose=0)
    predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
    predicted_class_label = predicted_class_index + 1
    print(f"\nPredicted class: {predicted_class_label}")    
