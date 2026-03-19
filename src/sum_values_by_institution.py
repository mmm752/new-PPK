import pandas as pd
import os

# Wczytaj najaktualniejszy PPK_master z output_csv
file_path = '/workspaces/new-PPK/output_csv/PPK_master.csv'

# Czytaj plik z separatorem średnika
df = pd.read_csv(file_path, sep=';')

# Wyodrębnij rok z kolumny 'quarter' (np. 4Q25 -> 2025, 4Q24 -> 2024)
df['rok'] = '20' + df['quarter'].str[-2:]

# Zsumuj wartosc_pln dla każdej instytucji
sums_by_institution = df.groupby('instytucja')['wartosc_pln'].sum().sort_values(ascending=False)

print("=" * 80)
print("SUMY WARTOŚCI PLN DLA WSZYSTKICH INSTYTUCJI")
print("=" * 80)
print(f"\nBaza danych: {os.path.basename(file_path)}")
print(f"\n{'INSTYTUCJA':<45} {'SUMA WARTOŚĆ PLN':>20}")
print("-" * 80)

for institution, total_value in sums_by_institution.items():
    print(f"{institution:<45} {total_value:>20,.2f}")

print("-" * 80)
print(f"{'RAZEM:':<45} {sums_by_institution.sum():>20,.2f}")
print("=" * 80)

# Zsumuj wartosc_pln dla każdej instytucji i roku
print("\n" + "=" * 80)
print("SUMY WARTOŚCI PLN DLA WSZYSTKICH INSTYTUCJI - PODZIAŁ NA LATA")
print("=" * 80)

sums_by_institution_year = df.groupby(['instytucja', 'rok'])['wartosc_pln'].sum().unstack(fill_value=0)

# Posortuj po najnowszych latach (od prawej do lewej) i sumo końcowej
sums_by_institution_year['RAZEM'] = sums_by_institution_year.sum(axis=1)
sums_by_institution_year = sums_by_institution_year.sort_values('RAZEM', ascending=False)

# Wydrukuj tabelę
print(f"\n{'INSTYTUCJA':<45}", end='')
for rok in sorted(sums_by_institution_year.columns[:-1]):  # bez RAZEM
    print(f"{rok:>15}", end='')
print(f"{'RAZEM':>20}")
print("-" * 100)

for institution in sums_by_institution_year.index:
    print(f"{institution:<45}", end='')
    for rok in sorted(sums_by_institution_year.columns[:-1]):  # bez RAZEM
        wartość = sums_by_institution_year.loc[institution, rok]
        if wartość > 0:
            print(f"{wartość:>15,.0f}", end='')
        else:
            print(f"{'':>15}", end='')
    print(f"{sums_by_institution_year.loc[institution, 'RAZEM']:>20,.0f}")

print("-" * 100)
print(f"{'RAZEM:':<45}", end='')
for rok in sorted(sums_by_institution_year.columns[:-1]):  # bez RAZEM
    total = sums_by_institution_year[rok].sum()
    print(f"{total:>15,.0f}", end='')
print(f"{sums_by_institution_year['RAZEM'].sum():>20,.0f}")
print("=" * 100)

# Zapisz wyniki do pliku CSV - całość
output_file = '/workspaces/new-PPK/output_csv/PPK_instytucja_sumy_wartosc_pln.csv'
output_df = pd.DataFrame({
    'instytucja': sums_by_institution.index,
    'suma_wartosc_pln': sums_by_institution.values
}).reset_index(drop=True)

output_df.to_csv(output_file, sep=';', index=False, decimal=',')
print(f"\nWyniki (całość) zapisane do: {output_file}")

# Zapisz wyniki do pliku CSV - podział na lata
output_file_years = '/workspaces/new-PPK/output_csv/PPK_instytucja_sumy_wartosc_pln_by_year.csv'
sums_by_institution_year.to_csv(output_file_years, sep=';', decimal=',')
print(f"Wyniki (podział na lata) zapisane do: {output_file_years}")

# Dodaj statystyki
print("\n" + "=" * 80)
print("STATYSTYKI")
print("=" * 80)
print(f"Liczba instytucji: {len(sums_by_institution)}")
print(f"Liczba lat w danych: {sorted(sums_by_institution_year.columns[:-1])}")
print(f"Średnia wartość na instytucję: {sums_by_institution.mean():,.2f} PLN")
print(f"Mediana: {sums_by_institution.median():,.2f} PLN")
print(f"Min: {sums_by_institution.min():,.2f} PLN ({sums_by_institution.idxmin()})")
print(f"Max: {sums_by_institution.max():,.2f} PLN ({sums_by_institution.idxmax()})")
