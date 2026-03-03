from __future__ import annotations

import argparse
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import shutil

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


def format_number(value: float) -> str:
    dec = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(dec, "f").rstrip("0").rstrip(".")
    return text or "0"


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


def sync_master_to_knf(master_path: Path, knf_path: Path, out_path: Path) -> tuple[int, int]:
    master = pd.read_csv(master_path, sep=";", dtype=str)
    knf_long = load_knf_long(knf_path)

    work = master.copy()
    work["_inst_raw"] = work.get("instytucja", "").fillna("").astype(str).str.strip()
    work["_inst"] = work["_inst_raw"].replace(NAME_MAP)
    work["_quarter"] = "4Q" + work.get("data", "").astype(str).str.slice(2, 4)
    work["_wartosc_num"] = parse_value_series(work.get("wartosc_pln", ""))

    scaled_pairs = 0
    added_pairs = 0

    for row in knf_long.itertuples(index=False):
        inst = row.instytucja
        quarter = row.quarter
        target = float(row.wartosc_knf) if pd.notna(row.wartosc_knf) else 0.0

        mask = (work["_inst"] == inst) & (work["_quarter"] == quarter)
        idx = work.index[mask]

        if len(idx) == 0:
            continue

        current = work.loc[idx, "_wartosc_num"].fillna(0)
        current_sum = float(current.sum())

        if current_sum == 0:
            largest_idx = idx[0]
            work.loc[idx, "_wartosc_num"] = 0.0
            work.loc[largest_idx, "_wartosc_num"] = target
            scaled_pairs += 1
            continue

        factor = target / current_sum
        scaled = current * factor

        if not scaled.empty:
            largest_local = scaled.abs().idxmax()
            residual = target - float(scaled.sum())
            scaled.loc[largest_local] = scaled.loc[largest_local] + residual

        work.loc[idx, "_wartosc_num"] = scaled
        scaled_pairs += 1

    work["wartosc_pln"] = work["_wartosc_num"].fillna(0).map(format_number)

    fund_total = (
        work.groupby(["instytucja", "fundusz", "data"], dropna=False)["_wartosc_num"]
        .sum()
        .rename("_fund_total")
    )
    work = work.merge(
        fund_total.reset_index(),
        on=["instytucja", "fundusz", "data"],
        how="left",
    )
    work["fund_total_pln"] = work["_fund_total"].fillna(0).map(format_number)
    work["fund_pct"] = work.apply(
        lambda r: "0"
        if pd.isna(r["_fund_total"]) or float(r["_fund_total"]) == 0
        else format_number(float(r["_wartosc_num"]) / float(r["_fund_total"])),
        axis=1,
    )

    drop_cols = [c for c in ["_inst_raw", "_inst", "_quarter", "_wartosc_num", "_fund_total"] if c in work.columns]
    work = work.drop(columns=drop_cols)

    columns = list(master.columns)
    for col in columns:
        if col not in work.columns:
            work[col] = ""
    work = work[columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
    return scaled_pairs, added_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronizacja wartosc_pln PPK_master do agregatów KNF")
    parser.add_argument("--master", default="clear/PPK_master.csv")
    parser.add_argument("--knf", default="clear/knf_reference.csv")
    parser.add_argument("--out", default="clear/PPK_master.csv")
    args = parser.parse_args()

    master_path = Path(args.master)
    out_path = Path(args.out)
    knf_path = Path(args.knf)

    if out_path.resolve() == master_path.resolve() and master_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = master_path.with_name(f"{master_path.stem}.backup_before_sync_{ts}{master_path.suffix}")
        shutil.copy2(master_path, backup)
        print(f"Backup: {backup}")

    scaled_pairs, added_pairs = sync_master_to_knf(master_path, knf_path, out_path)
    print(f"Zapisano: {out_path}")
    print(f"Przeskalowane pary instytucja/kwartał: {scaled_pairs}")
    print(f"Dodane pary instytucja/kwartał (syntetyczne): {added_pairs}")


if __name__ == "__main__":
    main()
