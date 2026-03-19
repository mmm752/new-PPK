# new-PPK

Repozytorium do budowy zbiorów PPK z danych surowych kwartalnych.

## Szybki workflow

1. Wrzuć pliki źródłowe do odpowiednich katalogów `raw_4Q22`, `raw_4Q23`, `raw_4Q24`, `raw_4Q25`.
2. Upewnij się, że w `clear` są tylko pliki referencyjne wymagane przez pipeline.
3. Uruchom `python src/main.py`.
4. Gotowe wyniki pojawią się w `output_csv`.
5. Pliki archiwalne i pomocnicze są poza głównym workflow w `archive` oraz `tools`.

## Minimalny układ katalogów

- `raw_4Q**` - wejścia kwartalne
- `clear` - wejścia pomocnicze dla pipeline
- `output_csv` - wszystkie główne wyniki generowane przez pipeline
- `archive` - starsze raporty, analizy i historyczne artefakty
- `tools` - pomocnicze skrypty poza głównym przebiegiem `main`

## Co musi zostać w clear

- `knf_reference.csv`
- `manual_shares_overrides.csv`

## Brak ilość_sztuk (liczba_sztuk)

Jeżeli w danych źródłowych brakuje liczba_sztuk (albo liczba_sztuk = 0), pipeline uzupełnia je z jednego wspólnego pliku referencyjnego:

- clear/manual_shares_overrides.csv

Uzupełnienie działa dla wszystkich instytucji podczas uruchomienia python src/main.py.

### Jak dodać brakujące sztuki

1. Otwórz clear/manual_shares_overrides.csv.
2. Dodaj wiersz z danymi pozycji, której brakuje liczba_sztuk.
3. Uzupełnij pola:
	- quarter (np. 4Q25)
	- instytucja (dokładnie jak w danych wejściowych)
	- fundusz (dokładnie jak w danych wejściowych)
	- emitent (dokładnie jak w kolumnie emitent)
	- shares_no (wartość brakującej liczba_sztuk)
4. Uruchom ponownie: python src/main.py.
5. Sprawdź wynik w output_csv/PPK_master.csv oraz output_csv/PPK_4Qxx.csv.

### Przykład wiersza

quarter;instytucja;fundusz;emitent;shares_no
4Q25;Pocztylion;PPK Pocztylion 2055 DFE;AB S.A.;1992,16

### Ważne zasady

- Dopasowanie działa po kluczu: quarter + instytucja + fundusz + emitent.
- Nazwy muszą być zgodne z danymi (najlepiej kopiować wartości 1:1).
- Liczba_sztuk jest uzupełniana tylko tam, gdzie była pusta lub równa 0.
- Jeżeli dodasz wiele identycznych rekordów klucza, shares_no zostaną zsumowane.
- Dla shares_no preferowany zapis to liczba z przecinkiem dziesiętnym, np. 1234,56.

## Uruchomienie

```bash
python src/main.py
```