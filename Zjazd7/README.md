# Raport z testów

Modele były testowane na przewidywaniu temperatury na 10 kolejnych godzin przy użyciu danych z 30 poprzednich. 
Model LSTM wypadł lepiej w testach. Jego prognozy wyglądają bardzo prawdopodobnie, znacznie bardziej naturanie od modelu w pełni połączonego.

## Przykładowe wywołania:
### Własny komputer
#### Model LSTM
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
32                |32                |units
0.001             |0.001             |lr

Trial 1 Complete [00h 00m 50s]
val_loss: 0.06290864199399948

Best val_loss So Far: 0.06290864199399948
Total elapsed time: 00h 00m 50s


Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
64                |32                |units
0.0001            |0.001             |lr

Trial 2 Complete [00h 01m 03s]
val_loss: 0.07050204277038574

Best val_loss So Far: 0.06290864199399948
Total elapsed time: 00h 01m 53s


Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
128               |32                |units
0.001             |0.001             |lr

Trial 3 Complete [00h 01m 38s]
val_loss: 0.06260057538747787

Best val_loss So Far: 0.06260057538747787
Total elapsed time: 00h 03m 31s

#### Model w pełni połączony
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
64                |64                |units
0.0001            |0.0001            |lr

Trial 1 Complete [00h 00m 10s]
val_loss: 0.08293090015649796

Best val_loss So Far: 0.08293090015649796
Total elapsed time: 00h 00m 10s


Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
128               |64                |units
0.0001            |0.0001            |lr

Trial 2 Complete [00h 00m 10s]
val_loss: 0.07879829406738281

Best val_loss So Far: 0.07879829406738281
Total elapsed time: 00h 00m 21s


Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
32                |128               |units
0.0001            |0.0001            |lr

Trial 3 Complete [00h 00m 10s]
val_loss: 0.08791526407003403

Best val_loss So Far: 0.07879829406738281
Total elapsed time: 00h 00m 31s

### Zdalnie
#### Model LSTM
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
32                |32                |units
0.001             |0.001             |lr

Trial 1 Complete [00h 02m 15s]
val_loss: 0.07159329950809479

Best val_loss So Far: 0.07159329950809479
Total elapsed time: 00h 02m 15s


Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
32                |32                |units
0.0001            |0.001             |lr

Trial 2 Complete [00h 02m 06s]
val_loss: 0.08640605211257935

Best val_loss So Far: 0.07159329950809479
Total elapsed time: 00h 04m 21s


Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
96                |32                |units
0.0001            |0.001             |lr

Trial 3 Complete [00h 02m 08s]
val_loss: 0.07633654028177261

Best val_loss So Far: 0.07159329950809479
Total elapsed time: 00h 06m 29s

#### Model w pełni połączony
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
128               |128               |units
0.0001            |0.0001            |lr

Trial 1 Complete [00h 01m 02s]
val_loss: 0.08701088279485703

Best val_loss So Far: 0.08701088279485703
Total elapsed time: 00h 01m 02s


Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
64                |128               |units
0.001             |0.0001            |lr
Trial 2 Complete [00h 01m 00s]
val_loss: 0.07793623954057693

Best val_loss So Far: 0.07793623954057693
Total elapsed time: 00h 02m 02s


Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
96                |64                |units
0.0001            |0.001             |lr

Trial 3 Complete [00h 01m 01s]
val_loss: 0.08974068611860275

Best val_loss So Far: 0.07793623954057693
Total elapsed time: 00h 03m 03s
