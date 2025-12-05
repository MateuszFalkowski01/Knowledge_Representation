# Podsumowanie

W projekcie ulepszyłem model z poprzednich zajęć za pomocą optymalizacji hiperparametrów przy użyciu Keras Tuner.

## Normalizacja

Normalizacja danych zdecydowanie poprawiła wyniki większość prób miała skuteczność powyżej 80%.

## Optymalizacja hiperparametrów

Baseline: 0.8333

Do optymalizacji wybrałem ilość neuronów pierwszej warstwy, współczynnik zapominania i tempo uczenia.

Co ciekawe najlepsze wyniki dla zadanego zbioru miały bardzo niski lub zerowy dropout i wysokie tempo uczenia.

# Przykładowe wyniki:

Found optimal hyperparameters:

* Units: 32
* Dropout: 0.0
* Learning Rate: 0.01

Best Model Evaluation: Loss: 0.0600, Accuracy: 0.9815

Found optimal hyperparameters:

* Units: 32
* Dropout: 0.2
* Learning Rate: 0.01

Best Model Evaluation: Loss: 0.2314, Accuracy: 0.9630

Found optimal hyperparameters:

* Units: 96
* Dropout: 0.0
* Learning Rate: 0.01

Best Model Evaluation: Loss: 0.0018, Accuracy: 1.0000



