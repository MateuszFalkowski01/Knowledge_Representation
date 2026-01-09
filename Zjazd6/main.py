import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, losses, Model
import os
from pathlib import Path

IMG_DIR = "Images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
LATENT_DIM = 2
EPOCHS = 400
            
            
train_ds = tf.keras.utils.image_dataset_from_directory(
    IMG_DIR,
    validation_split=0.2,
    subset="training",
    seed=2137,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    color_mode='rgb',
)
            
val_ds = tf.keras.utils.image_dataset_from_directory(
    IMG_DIR,
    validation_split=0.2,
    subset="validation",
    seed=2137,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    color_mode='rgb',
)

def process_data(x):
    # (Batch, Height, Width, Channels)
    x = tf.ensure_shape(x, [None, *IMG_SIZE, 3])
    x = x / 255.0
    return x, x

train_ds = train_ds.map(process_data)
val_ds = val_ds.map(process_data)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def apply_aug(img, target):
    aug_img = data_augmentation(img, training=True)
    return aug_img, aug_img

train_ds = train_ds.map(apply_aug)

# Optymalizacja wydajności GPU
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def save(img, result="obrazek.png"):
    plt.figure()
    plt.imshow(img)
    plt.title(result)
    plt.axis('off')
    plt.savefig(result)
    plt.close()

class AutoencoderCNN(Model):
    def __init__(self, latent_dim=LATENT_DIM, encoder0=None, decoder0=None):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ]) if encoder0 is None else encoder0

        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(16*16*128, activation='relu'),
            layers.Reshape((16, 16, 128)),
            layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv2D(3, 3, padding='same', activation='sigmoid')
        ]) if decoder0 is None else decoder0

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

try:
    encoder = tf.keras.models.load_model("model_encoder.keras")
    decoder = tf.keras.models.load_model("model_decoder.keras")
    do_not_fit = True
except Exception as e:
    encoder = decoder = None
    do_not_fit = False
    print("Nie znaleziono zapisanego modelu, rozpoczynam trening.")

autoencoder = AutoencoderCNN(LATENT_DIM, encoder0=encoder, decoder0=decoder)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

if not do_not_fit:
    autoencoder.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    autoencoder.encoder.save("model_encoder.keras")
    autoencoder.decoder.save("model_decoder.keras")

# Pobranie jednego batcha z walidacji do testów
for batch in val_ds.take(1):
    x_test_batch, _ = batch
    x_test_batch = x_test_batch.numpy()

out_name = "Output"

if not os.path.exists(out_name):
    os.makedirs(out_name)

limit = min(len(x_test_batch), 5)
for i in range(limit):
    save(x_test_batch[i], result=f"{out_name}/original_{i}.png")
    reconstructed = autoencoder.predict(x_test_batch[i:i+1], verbose=0) #i:i+1 - zachowanie wymiaru batcha
    save(reconstructed[0], result=f"{out_name}/reconstructed_{i}.png")
