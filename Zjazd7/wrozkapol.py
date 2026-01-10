import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def add_harmonics(df, column_name):
    values = df[column_name].values
    fft = np.fft.fft(values)
    df['amplitude'] = np.abs(fft)
    df['phase'] = np.angle(fft)
    return df

class TimeSeriesPatcher:
    def __init__(self, history_size, target_size):
        self.history_size = history_size
        self.target_size = target_size

    def create_sequences(self, data):
        x, y = [], []
        for i in range(len(data) - self.history_size - self.target_size + 1):
            x.append(data[i:(i + self.history_size)])
            y.append(data[(i + self.history_size):(i + self.history_size + self.target_size), 0])
        return np.array(x).astype(np.float32), np.array(y).astype(np.float32)

def build_model(hp, input_shape, n_output, model_type='LSTM'):
    model = tf.keras.Sequential()
    if model_type == 'LSTM':
        model.add(tf.keras.layers.LSTM(units=hp.Int('units', 32, 128, 32), input_shape=input_shape))
    else:
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        model.add(tf.keras.layers.Dense(units=hp.Int('units', 32, 128, 32), activation='relu'))
    
    model.add(tf.keras.layers.Dense(n_output))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('lr', [1e-3, 1e-4])), loss='mse')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, required=True)
    parser.add_argument('--column', type=str, default='T (degC)')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--result', type=str, default='yourluckynumbers.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.history)
    if 'Date Time' in df.columns:
        df = df[5::6]
    
    df = add_harmonics(df, args.column)
    features_names = [args.column, 'amplitude', 'phase']
    features = df[features_names].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    history_size = 30
    patcher = TimeSeriesPatcher(history_size, args.n)
    X, y = patcher.create_sequences(scaled_data)

    last_history = df[args.column].values[-50:] # Do wykresu: ostatnie 50 pkt historycznych
    results = {}
    scores = {} # Słownik na wyniki MAE

    for m_type in ['LSTM', 'Dense']:
        print(f"\n--- Optymalizacja: {m_type} ---")
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, (X.shape[1], X.shape[2]), args.n, m_type),
            objective='val_loss', max_trials=3, overwrite=True,
            directory='tuner_results', project_name=f'wrozka_{m_type}'
        )
        tuner.search(X, y, epochs=10, validation_split=0.2, verbose=1)
        
        best_trial = tuner.oracle.get_best_trials(1)[0]
        scores[m_type] = best_trial.score # val_loss (MSE)
        
        model = tuner.get_best_models(num_models=1)[0]
        
        last_window = scaled_data[-history_size:].reshape(1, history_size, X.shape[2])
        pred_scaled = model.predict(last_window, verbose=0)
        
        # Inwersja skalowania
        dummy = np.zeros((args.n, len(features_names)))
        dummy[:, 0] = pred_scaled[0]
        results[m_type] = scaler.inverse_transform(dummy)[:, 0]

    
    best_m = 'LSTM' if scores['LSTM'] < scores['Dense'] else 'Dense'
    print(f"\nZwycięzca: {best_m} (Best val_loss: {scores[best_m]:.4f})")
    
    pd.DataFrame(results[best_m], columns=['Prediction']).to_csv(args.result, index=False)
    print(f"Predykcje modelu {best_m} zapisano w {args.result}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(50), last_history, label='Historia', color='black', linewidth=2)
    
    future_range = range(50, 50 + args.n)
    plt.plot(future_range, results['LSTM'], 'o--', label='Predykcja LSTM', color='blue')
    plt.plot(future_range, results['Dense'], 's--', label='Predykcja Dense', color='orange')
    
    plt.title(f"Prognoza dla kolumny: {args.column}")
    plt.xlabel("Czas (kolejne próbki)")
    plt.ylabel("Wartość")
    plt.legend()
    plt.grid(True)
    
    print(f"\nPokazuję wykres...")
    plt.show()

if __name__ == "__main__":
    main()