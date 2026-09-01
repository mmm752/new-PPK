from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.main import build_master_dataset, parse_pzu_excel


def test_parse_pzu_excel_prefers_sheet_with_expected_headers(tmp_path):
    workbook_path = tmp_path / "pzu_2024-12-31.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame([["summary", "ignore me"]]).to_excel(writer, sheet_name="Summary", index=False, header=False)
        pd.DataFrame(
            [
                ["nazwa subfunduszu", "typ instrumentu", "emitent", "kod isin instrumentu"],
                ["PPK inPZU 2024", "Akcje", "PZU", "PLPZU0000011"],
            ]
        ).to_excel(writer, sheet_name="Data", index=False, header=False)

    df = parse_pzu_excel(str(workbook_path))

    assert not df.empty
    assert "PPK inPZU 2024" in df["fundusz"].astype(str).tolist()


def test_build_master_dataset_matches_goldman_sachs_and_santander_pairs():
    prev = pd.DataFrame(
        [
            {
                "quarter": "2Q25",
                "instytucja": "Erste",
                "fundusz": "Santander PPK 2025",
                "DATA_fundusz": "2025",
                "emitent": "SANTANDER BANK POLSKA S.A.",
                "isin": "PLBZ00000044",
                "waluta": "PLN",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "SPL PW Equity",
                "liczba_sztuk": "100",
                "wartosc_pln": "1000",
            },
            {
                "quarter": "2Q25",
                "instytucja": "ING",
                "fundusz": "Goldman Sachs Emerytura 2025",
                "DATA_fundusz": "2025",
                "emitent": "KGHM POLSKA MIEDŹ S.A.",
                "isin": "PLKGHM000017",
                "waluta": "PLN",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "KGH PW Equity",
                "liczba_sztuk": "50",
                "wartosc_pln": "500",
            },
        ]
    )
    curr = pd.DataFrame(
        [
            {
                "quarter": "2Q26",
                "instytucja": "Erste",
                "fundusz": "Erste PPK 2025",
                "DATA_fundusz": "2025",
                "emitent": "SANTANDER BANK POLSKA S.A.",
                "isin": "PLBZ00000044",
                "waluta": "PLN",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "SPL PW Equity",
                "liczba_sztuk": "110",
                "wartosc_pln": "1100",
            },
            {
                "quarter": "2Q26",
                "instytucja": "ING",
                "fundusz": "ING Emerytura 2025",
                "DATA_fundusz": "2025",
                "emitent": "KGHM POLSKA MIEDŹ S.A.",
                "isin": "PLKGHM000017",
                "waluta": "PLN",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "KGH PW Equity",
                "liczba_sztuk": "55",
                "wartosc_pln": "550",
            },
        ]
    )

    master_df = build_master_dataset({"2Q25": prev, "2Q26": curr})
    row_ppk = master_df.loc[master_df["fundusz"] == "Erste PPK 2025"].iloc[0]
    row_em = master_df.loc[master_df["fundusz"] == "ING Emerytura 2025"].iloc[0]

    assert row_ppk["liczba_sztuk_chg"] == "10"
    assert row_ppk["wartosc_pln_chg"] == "100"
    assert row_em["liczba_sztuk_chg"] == "5"
    assert row_em["wartosc_pln_chg"] == "50"
    assert row_ppk["Institutions_actual"] == "Erste"
    assert row_em["Institutions_actual"] == "ING"


def test_build_master_dataset_institutions_actual_maps_santander_and_goldman():
    df = pd.DataFrame(
        [
            {
                "quarter": "2Q26",
                "data": "30.06.2026",
                "instytucja": "Santander TFI S.A.",
                "fundusz": "Santander PPK 2025",
                "DATA_fundusz": "2025",
                "typ_aktywa": "Akcje",
                "emitent": "TEST",
                "isin": "PLTEST000001",
                "waluta": "PLN",
                "liczba_sztuk": "100",
                "wartosc_pln": "1000",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "TEST EQ",
            },
            {
                "quarter": "2Q26",
                "data": "30.06.2026",
                "instytucja": "Goldman Sachs TFI S.A.",
                "fundusz": "Goldman Sachs Emerytura 2025",
                "DATA_fundusz": "2025",
                "typ_aktywa": "Akcje",
                "emitent": "TEST2",
                "isin": "PLTEST000002",
                "waluta": "PLN",
                "liczba_sztuk": "200",
                "wartosc_pln": "2000",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "TEST EQ2",
            },
            {
                "quarter": "2Q26",
                "data": "30.06.2026",
                "instytucja": "Other TFI S.A.",
                "fundusz": "Other PPK 2025",
                "DATA_fundusz": "2025",
                "typ_aktywa": "Akcje",
                "emitent": "TEST3",
                "isin": "PLTEST000003",
                "waluta": "PLN",
                "liczba_sztuk": "300",
                "wartosc_pln": "3000",
                "TYP_aktywo_std": "akcje",
                "equity_nazwa": "TEST EQ3",
            },
        ]
    )

    master_df = build_master_dataset({"2Q26": df})
    actual = master_df.set_index("instytucja")["Institutions_actual"].to_dict()

    assert actual["Santander TFI S.A."] == "Erste"
    assert actual["Goldman Sachs TFI S.A."] == "ING"
    assert actual["Other TFI S.A."] == "Other TFI S.A."
