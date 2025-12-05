# Opis wyników

W projekcie porównałem dwa modele sieci neuronowych typu Sequential, które miały za zadanie klasyfikować wina na 3 kategorie w zbiorze Wine z UCI.

## Model prosty

- Dwie warstwy: 64 i 32 neurony
- Aktywacja: ReLU
- Wyjście: Softmax

## Model rozbudowany

- Trzy warstwy: 128 i 64 neurony
- Aktywacja: ELU
- Warstwa Dropout: 0.3

## Wyniki trenowania

W większości przypadków model prosty uczył się i popełniał mniej błędów:

Simple Model:  Loss: 0.9730, Accuracy: 0.8333
Complex Model: Loss: 1.2565, Accuracy: 0.8148

Jednakże zdarzały się próby gdzie wyniki były podobne, ale model rozbudowany dalej miał większą stratę:

Simple Model:  Loss: 0.5471, Accuracy: 0.8148
Complex Model: Loss: 0.7890, Accuracy: 0.8148

## Wnioski

Lepszym modelem okazał się **model prosty**, prawdopodobnie dlatego, że zbiór danych jest mały i łatwy. Wysoki dropout może również przeszkadzać **modelowi rozbudowanemu** w uczeniu na takim zbiorze.
