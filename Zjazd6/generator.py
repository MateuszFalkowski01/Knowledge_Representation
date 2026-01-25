import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import sys

MODEL_PATH = "model_decoder.keras"
OUTPUT_FILE = "generated_image.png"

def generate_image(z1, z2, z3, z4):
    try:
        decoder = tf.keras.models.load_model(MODEL_PATH)
        
        # model oczekuje batcha
        latent_vector = np.array([[z1, z2, z3, z4]], dtype=np.float32)

        prediction = decoder.predict(latent_vector, verbose=0)
        image = prediction[0]
        
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Latent: [{z1}, {z2}]")
        plt.axis('off')
        plt.savefig(OUTPUT_FILE)
        print(f"Obraz zapisany jako {OUTPUT_FILE}")
        plt.show()

    except Exception as e:
        print(f"Błąd podczas generowania: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        try:
            val1 = float(sys.argv[1])
            val2 = float(sys.argv[2])
            val3 = float(sys.argv[3])
            val4 = float(sys.argv[4])
            generate_image(val1, val2, val3, val4)
        except ValueError:
            print("Argumenty muszą być liczbami (np. 0.5 -1.2)")
    else:
        print("Użycie: python generator.py <wartość1> <wartość2>")
        print("Przykład: python generator.py 1.5 -0.5")