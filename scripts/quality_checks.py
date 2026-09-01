#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "output_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREV = OUT_DIR / "PPK_2Q25.csv"
CURR = OUT_DIR / "PPK_2Q26.csv"

if not PREV.exists() or not CURR.exists():
    print("Brak plików PPK_2Q25.csv lub PPK_2Q26.csv w output_csv. Przerwanie.")
    sys.exit(2)

def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=';', dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"Błąd wczytywania {path}: {e}")
        raise

df_prev = read_csv(PREV)
df_curr = read_csv(CURR)

key_cols = [c for c in ["instytucja", "fundusz", "emitent", "isin", "waluta", "liczba_sztuk", "wartosc_pln"] if c in df_prev.columns]

def per_inst_counts(df: pd.DataFrame) -> pd.Series:
    if "instytucja" not in df.columns:
        return pd.Series(dtype=int)
    return df.groupby(df["instytucja"]).size()

counts_prev = per_inst_counts(df_prev)
counts_curr = per_inst_counts(df_curr)

all_inst = sorted(set(counts_prev.index) | set(counts_curr.index))

report_rows = []
for inst in all_inst:
    prev_n = int(counts_prev.get(inst, 0))
    curr_n = int(counts_curr.get(inst, 0))
    diff = curr_n - prev_n
    pct = None
    if prev_n != 0:
        pct = (diff / prev_n) * 100
    report_rows.append({
        "instytucja": inst,
        "count_2Q25": prev_n,
        "count_2Q26": curr_n,
        "diff_count": diff,
        "pct_change_count": pct,
    })

rep = pd.DataFrame(report_rows)

def stats_series(s: pd.Series):
    arr = s.values.astype(float)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=0))
    return {"mean": mean, "median": median, "std": std, "q1": q1, "q3": q3, "iqr": iqr}

stats_prev = stats_series(rep["count_2Q25"]) if not rep.empty else {}
stats_curr = stats_series(rep["count_2Q26"]) if not rep.empty else {}
stats_diff = stats_series(rep["diff_count"]) if not rep.empty else {}

def mark_outliers_counts(row):
    flags = []
    # IQR method for prev
    if stats_prev:
        lower = stats_prev["q1"] - 1.5 * stats_prev["iqr"]
        upper = stats_prev["q3"] + 1.5 * stats_prev["iqr"]
        if row["count_2Q25"] < lower or row["count_2Q25"] > upper:
            flags.append("prev_count_outlier")
    if stats_curr:
        lower = stats_curr["q1"] - 1.5 * stats_curr["iqr"]
        upper = stats_curr["q3"] + 1.5 * stats_curr["iqr"]
        if row["count_2Q26"] < lower or row["count_2Q26"] > upper:
            flags.append("curr_count_outlier")
    # diff outlier
    if stats_diff and stats_diff.get("std", 0) > 0:
        z = abs((row["diff_count"] - stats_diff["mean"]) / stats_diff["std"]) if stats_diff["std"] > 0 else 0
        if z > 3:
            flags.append("diff_z_outlier")
    # present only in one period
    if row["count_2Q25"] == 0 and row["count_2Q26"] > 0:
        flags.append("only_in_2Q26")
    if row["count_2Q26"] == 0 and row["count_2Q25"] > 0:
        flags.append("only_in_2Q25")
    # large percent change
    if row["pct_change_count"] is not None:
        if abs(row["pct_change_count"]) > 200:
            flags.append("large_pct_change")
    return ";".join(flags)

rep["flags"] = rep.apply(mark_outliers_counts, axis=1)

# duplicates per instytucja
def duplicates_info(df: pd.DataFrame):
    if df.empty:
        return {}
    dup_mask = df.duplicated()
    duped = df[dup_mask]
    if duped.empty:
        return {}
    return duped.groupby(duped["instytucja"]).size().to_dict()

dups_prev = duplicates_info(df_prev)
dups_curr = duplicates_info(df_curr)

rep["dups_2Q25"] = rep["instytucja"].map(lambda i: int(dups_prev.get(i, 0)))
rep["dups_2Q26"] = rep["instytucja"].map(lambda i: int(dups_curr.get(i, 0)))

# missing/NULL checks: count rows with empty important fields
important = [c for c in ["fundusz", "emitent", "isin", "liczba_sztuk", "wartosc_pln"] if c in df_prev.columns]

def missing_counts(df: pd.DataFrame):
    if df.empty:
        return {}
    masks = {}
    for col in important:
        masks[col] = df[col].astype(str).replace({"": np.nan, "nan": np.nan}).isna()
    any_missing = pd.DataFrame(masks).any(axis=1)
    out = df[any_missing].groupby(df["instytucja"]).size().to_dict()
    zero_val = df[df.get("wartosc_pln", "").astype(str).replace({"": np.nan, "nan": np.nan}).fillna(0).astype(float) == 0].groupby(df["instytucja"]).size().to_dict() if "wartosc_pln" in df.columns else {}
    return {"missing_any": out, "zero_wartosc": zero_val}

miss_prev = missing_counts(df_prev)
miss_curr = missing_counts(df_curr)

rep["missing_any_2Q25"] = rep["instytucja"].map(lambda i: int(miss_prev.get("missing_any", {}).get(i, 0)))
rep["missing_any_2Q26"] = rep["instytucja"].map(lambda i: int(miss_curr.get("missing_any", {}).get(i, 0)))

rep["zero_wartosc_2Q25"] = rep["instytucja"].map(lambda i: int(miss_prev.get("zero_wartosc", {}).get(i, 0)))
rep["zero_wartosc_2Q26"] = rep["instytucja"].map(lambda i: int(miss_curr.get("zero_wartosc", {}).get(i, 0)))

out_csv = OUT_DIR / "PPK_completeness_2Q25_2Q26.csv"
rep.to_csv(out_csv, sep=';', index=False, encoding='utf-8-sig')

print("Zapisano raport kompletności:", out_csv)
print("Statystyki - 2Q25:", stats_prev)
print("Statystyki - 2Q26:", stats_curr)
print("Statystyki - różnice:", stats_diff)

outliers = rep[rep['flags'].astype(bool)].sort_values('flags')
print("Liczba instytucji z flagami (potencjalne anomalie):", len(outliers))
if not outliers.empty:
    print(outliers[['instytucja','count_2Q25','count_2Q26','diff_count','pct_change_count','flags']].to_string(index=False))
