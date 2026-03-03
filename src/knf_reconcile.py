from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


NAME_MAP = {
    "ESALIENS TFI S.A.": "Esaliens TFI S.A.",
    "Millennium TFI S.A.": "MILLENNIUM TFI S.A.",
    "Nationale-Nederlanden": "Nationale-Nederlanden PTE S.A.",
    "PZU TFI S.A.": "TFI PZU SA",
}


def parse_value_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.replace(" ", "", regex=False)
    has_comma = text.str.contains(",", regex=False)
    normalized = text.where(~has_comma, text.str.replace(".", "", regex=False))
    normalized = normalized.str.replace(",", ".", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


def load_master_totals(master_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, sep=";", dtype=str)
    values = parse_value_series(master["wartosc_pln"])
    quarter = "4Q" + master["data"].astype(str).str.slice(2, 4)
    inst = master["instytucja"].astype(str).str.strip().replace(NAME_MAP)

    agg = (
        pd.DataFrame({"instytucja": inst, "quarter": quarter, "wartosc_master": values})
        .groupby(["instytucja", "quarter"], dropna=False)["wartosc_master"]
        .sum()
        .reset_index()
    )
    return agg


def load_knf_long(knf_path: Path) -> pd.DataFrame:
    knf = pd.read_csv(knf_path, sep=";", dtype=str)
    quarter_cols = [c for c in knf.columns if re.fullmatch(r"4Q\d{2}_knf", c)]
    quarter_cols = sorted(quarter_cols, key=lambda c: int(c[2:4]))
    if "instytucja" not in knf.columns:
        raise ValueError("Brakuje kolumny KNF: instytucja")
    if not quarter_cols:
        raise ValueError("Brakuje kolumn KNF w formacie 4QYY_knf")

    long = knf.melt(id_vars=["instytucja"], value_vars=quarter_cols, var_name="q", value_name="wartosc_knf")
    long["quarter"] = long["q"].str.replace("_knf", "", regex=False)
    long["wartosc_knf"] = parse_value_series(long["wartosc_knf"])
    return long[["instytucja", "quarter", "wartosc_knf"]]


def build_report(master_path: Path, knf_path: Path, threshold_pct: float) -> pd.DataFrame:
    master = load_master_totals(master_path)
    knf = load_knf_long(knf_path)

    report = knf.merge(master, on=["instytucja", "quarter"], how="left")
    report["wartosc_master"] = report["wartosc_master"].fillna(0)
    report["ratio_pct"] = (report["wartosc_master"] / report["wartosc_knf"]) * 100
    report["abs_dev_pct"] = (report["ratio_pct"] - 100).abs()
    report["diff_pln"] = report["wartosc_master"] - report["wartosc_knf"]
    report["status"] = report["abs_dev_pct"].apply(lambda x: "OK" if x <= threshold_pct else "ALERT")
    return report.sort_values(["status", "abs_dev_pct"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Porównanie KNF vs PPK_master z progiem odchylenia")
    parser.add_argument("--master", default="clear/PPK_master.csv")
    parser.add_argument("--knf", default="clear/knf_reference.csv")
    parser.add_argument("--out", default="clear/knf_reconciliation_report.csv")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    report = build_report(Path(args.master), Path(args.knf), args.threshold)
    report.to_csv(args.out, sep=";", index=False, encoding="utf-8-sig")

    alerts = report[report["status"] == "ALERT"]
    print(f"Zapisano raport: {args.out}")
    print(f"Wiersze ALERT (> {args.threshold}%): {len(alerts)} / {len(report)}")
    if not alerts.empty:
        print(alerts[["instytucja", "quarter", "wartosc_knf", "wartosc_master", "ratio_pct", "abs_dev_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
