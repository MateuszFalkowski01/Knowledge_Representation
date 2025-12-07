# Opis wyników

Oba modele uczą się prawidłowo. Konwolucyjny osiąga odrobinę lepsze wyniki.

Średnie wyniki modeli:

Bez generalizacji:
Model oparty o warstwy w pełni połączone:
Loss: ~0.4
Accuracy: ~84,5%
Model oparty o warstwy splotowe:
Loss: ~0.3
Accuracy: 88%


Z generalizacją:
Model oparty o warstwy w pełni połączone:
Loss: ~0.8
Accuracy: ~70%
Model oparty o warstwy splotowe:
Loss: ~0.6
Accuracy: 75%


Jak można było się spodziewać generalizacja danych za pomocą augmentacji powoduje spadek wyników, ale niestety nie udało się zmierzyć czy pomaga w predykcji obrazów spoza trenowanych zbiorów. Wszystkie modele przeważnie odczytywały obrazy z internetu poprawnie. Co ciekawe w przypadku koszulki z wzorkiem wszystkie zakwalifikowały go błednie jako płaszcz.
