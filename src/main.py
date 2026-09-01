import os
import glob
import re
import signal
import unicodedata
import multiprocessing as mp
import shutil
import tempfile
import subprocess
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
import PyPDF2
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover
    pdfminer_extract_text = None
try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None
try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None

from quarters import (
    parse_quarter_token,
    quarter_sort_key,
    prev_quarter_token,
    quarter_token_from_folder,
    quarter_end_date_from_folder,
    quarter_token_from_date,
)

OUTPUT_COLUMNS = [
    "data",
    "instytucja",
    "fundusz",
    "DATA_fundusz",
    "typ_aktywa",
    "emitent",
    "isin",
    "waluta",
    "liczba_sztuk",
    "wartosc_pln",
    "TYP_aktywo_std",
    "equity_nazwa",
]

EQUITY_FILE = "equity.xlsx"
ISIN_FILE = "isin.xlsx"

PDF_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
}


# -------------------------
# Helpers
# -------------------------

def normalize_header(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    return cleaned.replace(":", "")


def make_unique_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for header in headers:
        base = header or ""
        count = seen.get(base, 0) + 1
        seen[base] = count
        unique.append(f"{base}__{count}" if count > 1 else base)
    return unique


def find_column(df: pd.DataFrame, expected: str) -> Optional[str]:
    if expected in df.columns:
        return expected
    prefix = f"{expected}__"
    for col in df.columns:
        if col.startswith(prefix):
            return col
    return None


def parse_polish_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    text = text.replace(" ", "")
    if "," in text:
        text = text.replace(".", "")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def is_timestamp_like_amount(value) -> bool:
    text = safe_string(value).replace(" ", "")
    if not text:
        return False
    normalized = text.replace(",", ".")
    return bool(re.match(r"^20\d{9,}(?:\.\d+)?$", normalized))


def sanitize_wartosc_pln_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df is None or df.empty or "wartosc_pln" not in df.columns:
        return df, 0

    work = df.copy()
    raw = work["wartosc_pln"].astype(str).str.strip()
    numeric = raw.apply(parse_polish_number)

    timestamp_like = raw.apply(is_timestamp_like_amount)
    pocztylion_fwd = (
        work.get("instytucja", "").astype(str).str.strip().eq("Pocztylion")
        & work.get("emitent", "").astype(str).str.strip().str.match(r"^FWD[A-Z]{2}PL\d{8}$", na=False)
    )
    extreme_value = numeric.gt(1e9).fillna(False)

    anomaly_mask = timestamp_like | (pocztylion_fwd & extreme_value)
    removed = int(anomaly_mask.sum())
    if removed:
        work = work.loc[~anomaly_mask].copy()

    return work, removed


def format_decimal_comma(value) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, str) and value.strip() == "":
        return "nan"
    text = str(value).strip()
    if text.lower() == "nan":
        return "nan"
    try:
        number = Decimal(text)
    except (TypeError, ValueError):
        return text.replace(",", ".")

    # Zaokrąglij do 2 miejsc i usuń trailing zeros
    number = number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(number, "f").rstrip("0").rstrip(".")

    return text


def safe_string(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_typ_aktywa(value: str) -> str:
    text = safe_string(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    if text == "" or text == "nan":
        return ""

    # Check for derivatives FIRST, before checking for "akcj" keyword
    if "instrument" in text and "pochodn" in text:
        return "inne"
    if "reit" in text:
        return "akcje"
    
    if re.search(r"\bakcj", text):
        return "akcje"
    if "aktywa" in text and "udzia" in text:
        return "akcje"
    if re.search(r"\boblig", text) or "dłużn" in text or "dluzn" in text:
        return "obligacje"
    if "list zastawn" in text or "papier komerc" in text:
        return "obligacje"

    return "inne"


def _fix_mojibake_text(value: str) -> str:
    text = safe_string(value)
    if not text:
        return text
    if re.search(r"[ÃÄÅÂĹ]", text):
        try:
            return text.encode("cp1250").decode("utf-8")
        except Exception:
            return text
    return text


_PZU_TYPAKTYWA_FIXES = {
    "DĹ\x81UŻNE PAPIERY WARTOĹšCIOWE": "DŁUŻNE PAPIERY WARTOŚCIOWE",
    "TYTUĹ\x81Y UCZESTNICTWA": "TYTUŁY UCZESTNICTWA",
    "AKTYWA UDZIAĹ\x81OWE": "AKTYWA UDZIAŁOWE",
    "DĹ\x81UĹ»NE_PAPIERY": "DŁUŻNE_PAPIERY",
}


_PZU_EOP_2023_VALUES = {
    "2025": 524_672,
    "2030": 751_855,
    "2035": 926_151,
    "2040": 877_252,
    "2045": 691_647,
    "2050": 440_813,
    "2055": 260_705,
    "2060": 86_936,
    "2065": 6_141,
}


def normalize_pzu_fundusz(value: str) -> str:
    text = _fix_mojibake_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    match = re.search(r"(20\d{2})", text)
    if not match:
        return text
    year = match.group(1)
    if re.search(r"^\s*(?:PPK\s*inPZU|inPZU\s*PPK)\b", text, re.IGNORECASE):
        return f"PPK inPZU {year}"
    if re.search(r"^\s*inPZU\s*Puls\s*(Życia|Zycia)\b", text, re.IGNORECASE):
        return f"inPZU Puls Życia {year}"
    return text


def is_pzu_ppk_fund(value: str) -> bool:
    text = _fix_mojibake_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    return bool(
        re.search(
            r"^\s*(?:PPK\s*inPZU|inPZU\s*PPK|inPZU\s*Puls\s*Życia|inPZU\s*Puls\s*Zycia)\b.*(20\d{2})",
            text,
            re.IGNORECASE,
        )
    )


def fix_pzu_shifted_isin_waluta(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"emitent", "isin", "waluta"}
    if not required_cols.issubset(df.columns):
        return df

    isin_pattern = r"^[A-Z]{2}[A-Z0-9]{10}$"
    waluta_series = df["waluta"].astype(str).str.strip()
    isin_series = df["isin"].astype(str).str.strip()
    emitent_series = df["emitent"].astype(str).str.strip()

    waluta_is_isin = waluta_series.str.match(isin_pattern, na=False)
    isin_is_isin = isin_series.str.match(isin_pattern, na=False)

    move_mask = waluta_is_isin & ~isin_is_isin
    if move_mask.any():
        emitent_blank = emitent_series.str.lower().isin(["", "nan", "none", "0"])
        isin_candidate = isin_series.where(~isin_series.str.lower().isin(["", "nan", "none", "0"]), "")

        fill_emitent_mask = move_mask & emitent_blank & isin_candidate.ne("")
        df.loc[fill_emitent_mask, "emitent"] = isin_candidate[fill_emitent_mask]
        df.loc[move_mask, "isin"] = waluta_series[move_mask]

    # Jeżeli w walucie nadal jest kod ISIN, wyczyść to pole (to nie jest waluta instrumentu)
    df.loc[waluta_is_isin, "waluta"] = ""
    return df


def apply_pzu_eop_2023_correction(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Apply manual correction for PZU EOP 2023 values (ISIN PLPZU0000011)
    for PZU1_2023-12-31 source file.
    """
    file_name = os.path.basename(file_path).lower()
    if file_name != "pzu1_2023-12-31.xlsx":
        return df

    if df.empty:
        return df

    work = df.copy()
    if "DATA_fundusz" not in work.columns:
        work["DATA_fundusz"] = ""

    fund_prefix = "PPK inPZU"
    fundusz_series = work.get("fundusz", pd.Series("", index=work.index)).astype(str)
    if not fundusz_series.str.contains(r"^\s*PPK\s*inPZU\b", case=False, regex=True, na=False).any():
        fund_prefix = "inPZU Puls Życia"

    template = {col: "" for col in work.columns}
    if not work.empty:
        sample = work.iloc[0].to_dict()
        template.update({k: ("" if pd.isna(v) else v) for k, v in sample.items()})

    new_rows: List[Dict[str, Any]] = []

    target_years = set(_PZU_EOP_2023_VALUES.keys())
    year_series_norm = (
        work["DATA_fundusz"]
        .astype(str)
        .str.extract(r"(20\d{2})", expand=False)
        .fillna("")
    )
    year_mask_all = year_series_norm.isin(target_years)
    id_mask_all = (
        work.get("isin", pd.Series("", index=work.index)).astype(str).str.upper().str.strip().eq("PLPZU0000011")
        | work.get("emitent", pd.Series("", index=work.index)).astype(str).str.contains(r"\bPZU\b", case=False, regex=True, na=False)
        | work.get("nazwa_instrumentu", pd.Series("", index=work.index)).astype(str).str.contains(r"\bPZU\b", case=False, regex=True, na=False)
    )

    # Drop all existing PZU EOP rows for 2025-2065 and replace with one authoritative row per year.
    work = work.loc[~(year_mask_all & id_mask_all)].copy()

    for year, corrected_value in _PZU_EOP_2023_VALUES.items():
        row = dict(template)
        row["fundusz"] = f"{fund_prefix} {year}"
        row["DATA_fundusz"] = year
        row["typ_aktywa"] = "Akcje"
        row["emitent"] = "PZU"
        row["isin"] = "PLPZU0000011"
        row["waluta"] = "PLN"
        row["liczba_sztuk"] = ""
        row["wartosc_pln"] = corrected_value
        row["TYP_aktywo_std"] = "akcje"
        new_rows.append(row)

    if new_rows:
        work = pd.concat([work, pd.DataFrame(new_rows)], ignore_index=True)

    return work


def _normalize_equity_name(value: str) -> str:
    text = safe_string(value).lower()
    text = text.replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(".", "")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def _equity_name_variants(value: str) -> List[str]:
    normalized = _normalize_equity_name(value)
    if not normalized:
        return []

    variants = {normalized}
    tokens = normalized.split()

    legal_tail_tokens = {
        "sa",
        "spa",
        "se",
        "ag",
        "nv",
        "plc",
        "inc",
        "corp",
        "corporation",
        "holding",
        "holdings",
        "group",
        "co",
        "company",
        "ltd",
        "limited",
    }
    trimmed = tokens[:]
    while trimmed and trimmed[-1] in legal_tail_tokens:
        trimmed.pop()
    if trimmed:
        variants.add(" ".join(trimmed))

    removable_tokens = {"sa", "asi", "spa", "se", "ag", "nv", "plc"}
    compact = [token for token in tokens if token not in removable_tokens]
    if compact:
        variants.add(" ".join(compact))

    return sorted(variants, key=len, reverse=True)


def _best_ticker_from_name(emitent_value: str, equity_name_map: Dict[str, str]) -> str:
    if not equity_name_map:
        return ""

    for variant in _equity_name_variants(emitent_value):
        ticker = equity_name_map.get(variant)
        if ticker:
            return ticker
    return ""


def _best_ticker_from_alias_patterns(emitent_value: str) -> str:
    normalized = _normalize_equity_name(emitent_value)
    if not normalized:
        return ""
    compact = normalized.replace(" ", "")

    contains_aliases = [
        ("gielda papierow wartosciowych w warszawie", "GPW PW Equity"),
        ("famur", "GEA PW Equity"),
        ("cytokinet", "CYTK US Equity"),
        ("greenvolt", "GVOLT PL Equity"),
        ("hafnia", "HAFNI NO Equity"),
        ("karuna therapeutics", "KRTX US Equity"),
        ("tjx companies", "TJX US Equity"),
        ("cspx ishares vii", "CSPX LN Equity"),
        ("ishrs core s p 500 ucits etf usd acc", "CSPX LN Equity"),
        ("jastrzebska spolka weglowa", "JSW PW Equity"),
        ("livechat software", "TXT PW Equity"),
        ("polski koncern naftowy orlen", "PKN PW Equity"),
        ("alumetal", "AML PW Equity"),
        ("ciech", "CIE PW Equity"),
        ("m d c holdings", "MDC US Equity"),
        ("kernel holding", "KER PW Equity"),
        ("bank polska kasa opieki", "PEO PW Equity"),
        ("kghm polska miedz", "KGH PW Equity"),
    ]

    for marker, ticker in contains_aliases:
        marker_compact = marker.replace(" ", "")
        if marker in normalized or marker_compact in compact:
            return ticker

    return ""


def load_isin_mapping(base_dir: str) -> Dict[str, str]:
    """Wczytuje isin.xlsx i zwraca mapowanie equity_nazwa -> ISIN."""
    isin_path = os.path.join(base_dir, ISIN_FILE)
    if not os.path.exists(isin_path):
        return {}
    try:
        df_isin = pd.read_excel(isin_path, header=None, engine="openpyxl")
    except Exception:
        return {}
    isin_map: Dict[str, str] = {}
    for _, row in df_isin.iterrows():
        ticker = safe_string(row.iloc[0]).strip()
        isin = safe_string(row.iloc[1]).strip().upper()
        if ticker and isin and isin not in ("0", "", "NAN", "NONE"):
            isin_map[ticker] = isin
    return isin_map


def fill_missing_isin(df: pd.DataFrame, isin_map: Dict[str, str]) -> pd.DataFrame:
    """Uzupełnia brakujące kody ISIN (wartość 0) dla akcji na bazie equity_nazwa."""
    if not isin_map or "isin" not in df.columns or "equity_nazwa" not in df.columns or "TYP_aktywo_std" not in df.columns:
        return df
    mask = (
        df["TYP_aktywo_std"].eq("akcje")
        & df["isin"].astype(str).isin(["0", "", "nan"])
        & df["equity_nazwa"].notna()
        & ~df["equity_nazwa"].isin(["", "0"])
    )
    if mask.any():
        df = df.copy()
        df.loc[mask, "isin"] = df.loc[mask, "equity_nazwa"].map(isin_map).fillna(df.loc[mask, "isin"])
    return df


def load_equity_mapping(base_dir: str) -> tuple[Dict[str, str], Dict[str, str]]:
    equity_path = os.path.join(base_dir, EQUITY_FILE)
    if not os.path.exists(equity_path):
        return {}, {}
    try:
        df_equity = pd.read_excel(equity_path, engine="openpyxl")
    except Exception:
        return {}, {}

    if "id_isin" not in df_equity.columns or "ID" not in df_equity.columns:
        return {}, {}

    isin_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    invalid_tickers = {"", "0", "nan", "none"}
    for _, row in df_equity.iterrows():
        isin = safe_string(row.get("id_isin", "")).upper()
        ticker = safe_string(row.get("ID", ""))
        if isin and ticker and isin.lower() != "nan" and ticker.lower() not in invalid_tickers:
            isin_map[isin] = ticker
        if ticker and ticker.lower() not in invalid_tickers:
            for name_variant in _equity_name_variants(row.get("name", "")):
                if name_variant and name_variant.lower() != "nan":
                    name_map.setdefault(name_variant, ticker)
    return isin_map, name_map


def apply_equity_nazwa(
    df: pd.DataFrame,
    equity_map: Dict[str, str],
    equity_name_map: Dict[str, str],
) -> pd.DataFrame:
    if "equity_nazwa" not in df.columns:
        df["equity_nazwa"] = ""

    emitent_series_all = df.get("emitent", "").astype(str).str.strip()
    pattern_ticker_all = emitent_series_all.apply(_best_ticker_from_alias_patterns)

    typ_series = df.get("TYP_aktywo_std", "").astype(str).str.strip().str.lower()
    is_akcje = typ_series.eq("akcje")

    if is_akcje.any():
        isin_series = df.get("isin", "").astype(str).str.strip().str.upper()
        emitent_series = df.get("emitent", "").astype(str).str.strip()

        alias_map = {
            "gielda papierow wartosciowych w warszawie sa": "GPW PW Equity",
            "mci capital asi sa": "MCI PW Equity",
            "powszechna kasa oszczednosci bank polski sa": "PKO PW Equity",
            "pekao": "PEO PW Equity",
            "bank pekao sa": "PEO PW Equity",
            "bank pekao s a": "PEO PW Equity",
            "bank polska kasa opieki sa": "PEO PW Equity",
            "poznanska korporacja budowlana pekabex sa": "PBX PW Equity",
            "poznanska korporacja budowlana pekabex": "PBX PW Equity",
        }

        emitent_ticker = emitent_series.apply(lambda name: _best_ticker_from_name(name, equity_name_map))
        alias_ticker = emitent_series.apply(lambda name: alias_map.get(_normalize_equity_name(name), ""))
        pattern_ticker = emitent_series.apply(_best_ticker_from_alias_patterns)

        df.loc[is_akcje, "equity_nazwa"] = isin_series.map(equity_map)

        nn_mask = is_akcje & df.get("instytucja", "").astype(str).eq("Nationale-Nederlanden")
        if nn_mask.any():
            nn_missing = df.loc[nn_mask, "equity_nazwa"].isna() | df.loc[nn_mask, "equity_nazwa"].eq("")
            df.loc[nn_mask & nn_missing, "equity_nazwa"] = emitent_ticker

        pocztylion_mask = is_akcje & df.get("instytucja", "").astype(str).eq("Pocztylion")
        if pocztylion_mask.any():
            pocz_missing = df.loc[pocztylion_mask, "equity_nazwa"].isna() | df.loc[pocztylion_mask, "equity_nazwa"].eq("")
            df.loc[pocztylion_mask & pocz_missing, "equity_nazwa"] = emitent_ticker

        global_missing = is_akcje & (df["equity_nazwa"].isna() | df["equity_nazwa"].isin(["", "0"]))
        if global_missing.any():
            df.loc[global_missing, "equity_nazwa"] = emitent_ticker

        alias_missing = is_akcje & (df["equity_nazwa"].isna() | df["equity_nazwa"].isin(["", "0"]))
        if alias_missing.any():
            df.loc[alias_missing, "equity_nazwa"] = alias_ticker

        pattern_missing = is_akcje & (df["equity_nazwa"].isna() | df["equity_nazwa"].isin(["", "0"]))
        if pattern_missing.any():
            df.loc[pattern_missing, "equity_nazwa"] = pattern_ticker

        forced_name_override = is_akcje & pattern_ticker.ne("")
        if forced_name_override.any():
            df.loc[forced_name_override, "equity_nazwa"] = pattern_ticker

        bnp_mask = is_akcje & emitent_series.eq("BNP Paribas Bank Polska S.A.")
        df.loc[bnp_mask, "equity_nazwa"] = "BNPPPL PW Equity"

        atal_mask = is_akcje & emitent_series.apply(_normalize_equity_name).eq("atal sa")
        df.loc[atal_mask, "equity_nazwa"] = "1AT PW Equity"

        alphabet_mask = (
            is_akcje
            & emitent_series.eq("ALPHABET INC.")
            & isin_series.eq("US02079K1079")
        )
        df.loc[alphabet_mask, "equity_nazwa"] = "GOOGL US Equity"

        readly_mask = (
            is_akcje
            & emitent_series.eq("PLN READLY INTERNATIONAL AB")
            & isin_series.eq("SE0026599334")
        )
        df.loc[readly_mask, "equity_nazwa"] = "READ SS Equity"

        citigroup_lux_mask = is_akcje & isin_series.eq("LU2414210828")
        df.loc[citigroup_lux_mask, "equity_nazwa"] = "C US Equity"

        barrick_gold_mask = is_akcje & isin_series.eq("CA0679011084")
        df.loc[barrick_gold_mask, "equity_nazwa"] = "B US Equity"

        astrazeneca_mask = is_akcje & isin_series.eq("US0463531089")
        df.loc[astrazeneca_mask, "equity_nazwa"] = "AZN US Equity"

        blackrock_mask = is_akcje & isin_series.eq("US09247X1019")
        df.loc[blackrock_mask, "equity_nazwa"] = "BLK US Equity"

        infinera_mask = is_akcje & isin_series.eq("US45667G1031")
        df.loc[infinera_mask, "equity_nazwa"] = "INFN US Equity"

        df.loc[is_akcje, "equity_nazwa"] = df.loc[is_akcje, "equity_nazwa"].fillna("NA")

        na_equity_mask = df["equity_nazwa"].astype(str).str.strip().eq("NA")
        if na_equity_mask.any() and "TYP_aktywo_std" in df.columns:
            df.loc[na_equity_mask, "TYP_aktywo_std"] = "inne"

    recovery_mask = pattern_ticker_all.ne("") & df["equity_nazwa"].astype(str).str.strip().isin(["", "0", "NA", "nan"])
    if recovery_mask.any():
        df.loc[recovery_mask, "equity_nazwa"] = pattern_ticker_all

    if "TYP_aktywo_std" in df.columns:
        # Only upgrade rows already classified as equities.
        # Avoid turning debt/derivative holdings into akcje just because a ticker pattern matched.
        mapped_from_alias = (
            pattern_ticker_all.ne("")
            & df["equity_nazwa"].astype(str).str.strip().ne("")
            & typ_series.eq("akcje")
        )
        if mapped_from_alias.any():
            df.loc[mapped_from_alias, "TYP_aktywo_std"] = "akcje"

        # UNIQA fallback: if typ_aktywa says "Akcje" but ticker cannot be identified,
        # classify as "inne" to avoid keeping unresolved equities in akcje bucket.
        uniqa_unresolved = (
            df.get("instytucja", "").astype(str).str.strip().eq("UNIQA TFI S.A.")
            & df.get("typ_aktywa", "").astype(str).str.strip().str.lower().eq("akcje")
            & df["equity_nazwa"].astype(str).str.strip().isin(["", "0", "NA", "nan"])
        )
        if uniqa_unresolved.any():
            df.loc[uniqa_unresolved, "TYP_aktywo_std"] = "inne"

    return df


def load_fixed_shares_map(base_dir: str, relative_path: str) -> Dict[str, float]:
    fixed_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(fixed_path):
        return {}
    try:
        if fixed_path.lower().endswith((".xlsx", ".xls")):
            fixed_df = pd.read_excel(fixed_path, dtype=str)
        else:
            fixed_df = pd.read_csv(fixed_path, sep=";", dtype=str, keep_default_na=False)
    except Exception:
        return {}

    required_cols = {"quarter", "instytucja", "fundusz", "shares_no"}
    if not required_cols.issubset(fixed_df.columns):
        return {}

    emitent_col = None
    for candidate in ("emitent_full", "emitent"):
        if candidate in fixed_df.columns:
            emitent_col = candidate
            break
    if emitent_col is None:
        return {}

    fixed_df["shares_no_num"] = fixed_df["shares_no"].apply(parse_polish_number)
    fixed_df = fixed_df[fixed_df["shares_no_num"].notna()].copy()
    if fixed_df.empty:
        return {}

    fixed_df["fund_norm"] = fixed_df["fundusz"].apply(_normalize_equity_name)
    fixed_df["emitent_norm"] = fixed_df[emitent_col].apply(_normalize_equity_name)
    fixed_df["map_key"] = (
        fixed_df["quarter"].astype(str).str.strip()
        + "|"
        + fixed_df["instytucja"].astype(str).str.strip()
        + "|"
        + fixed_df["fund_norm"]
        + "|"
        + fixed_df["emitent_norm"]
    )

    aggregated = fixed_df.groupby("map_key", dropna=False)["shares_no_num"].sum(min_count=1)
    return aggregated.to_dict()


def load_manual_shares_map(base_dir: str) -> Dict[str, float]:
    return load_fixed_shares_map(base_dir, os.path.join("clear", "manual_shares_overrides.csv"))


def apply_manual_shares_map(
    df: pd.DataFrame,
    quarter_token: str,
    shares_map: Dict[str, float],
) -> pd.DataFrame:
    if not shares_map or df.empty:
        return df
    if "instytucja" not in df.columns:
        return df

    out = df.copy()
    instytucja_series = out.get("instytucja", "").astype(str).str.strip()
    out["fund_norm"] = out.get("fundusz", "").astype(str).apply(_normalize_equity_name)
    out["emitent_norm"] = out.get("emitent", "").astype(str).apply(_normalize_equity_name)
    out["map_key"] = (
        quarter_token.strip()
        + "|"
        + instytucja_series
        + "|"
        + out["fund_norm"]
        + "|"
        + out["emitent_norm"]
    )
    out["manual_shares_num"] = out["map_key"].map(shares_map)

    current_qty = out.get("liczba_sztuk", "").apply(parse_polish_number)
    fill_mask = out["manual_shares_num"].notna() & (current_qty.isna() | current_qty.eq(0))
    if fill_mask.any():
        out.loc[fill_mask, "liczba_sztuk"] = out.loc[fill_mask, "manual_shares_num"].apply(format_decimal_comma)

    out = out.drop(columns=["fund_norm", "emitent_norm", "map_key", "manual_shares_num"], errors="ignore")
    return out

def ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]
    
    # Convert data column to YYYY-MM-DD format if it's datetime
    if "data" in df.columns:
        def _normalize_date(value) -> Optional[str]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            text = str(value).strip()
            if text == "" or text.lower() == "nan":
                return None
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")

        df["data"] = df["data"].apply(_normalize_date)
    
    df["liczba_sztuk"] = df["liczba_sztuk"].apply(parse_polish_number)
    df["wartosc_pln"] = df["wartosc_pln"].apply(parse_polish_number)

    # Drop rows not assigned to any fund
    if "fundusz" in df.columns:
        fundusz_series = df["fundusz"]
        mask = (
            fundusz_series.notna()
            & fundusz_series.astype(str).str.strip().ne("")
            & fundusz_series.astype(str).str.strip().str.lower().ne("nan")
        )
        df = df[mask]

    # Extract year from fundusz into DATA_fundusz
    if "fundusz" in df.columns:
        df["DATA_fundusz"] = (
            df["fundusz"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )

    # Normalize typ_aktywa into a standard bucket
    if "typ_aktywa" in df.columns:
        df["TYP_aktywo_std"] = df["typ_aktywa"].apply(normalize_typ_aktywa)

    df = df.replace("", pd.NA)
    df = df.fillna("nan")
    return df


def _valid_group_mask(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in group_cols:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[col].astype(str)
        mask &= (
            series.str.strip().ne("")
            & series.str.strip().str.lower().ne("nan")
        )
    return mask


def build_akcje_share(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if "wartosc_pln" not in df.columns:
        return pd.DataFrame(columns=group_cols + ["wartosc_pln_total", "wartosc_pln_akcje", "udzial_akcje_pct"])

    mask = _valid_group_mask(df, group_cols)
    if not mask.any():
        return pd.DataFrame(columns=group_cols + ["wartosc_pln_total", "wartosc_pln_akcje", "udzial_akcje_pct"])

    work_df = df.loc[mask, group_cols + ["wartosc_pln", "TYP_aktywo_std"]].copy()
    work_df["wartosc_pln_num"] = work_df["wartosc_pln"].apply(parse_polish_number)
    work_df["is_akcje"] = work_df["TYP_aktywo_std"].astype(str).str.strip().str.lower().eq("akcje")

    total_df = (
        work_df.groupby(group_cols, dropna=False)["wartosc_pln_num"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"wartosc_pln_num": "wartosc_pln_total"})
    )
    akcje_df = (
        work_df[work_df["is_akcje"]]
        .groupby(group_cols, dropna=False)["wartosc_pln_num"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"wartosc_pln_num": "wartosc_pln_akcje"})
    )

    merged = total_df.merge(akcje_df, on=group_cols, how="left")
    merged["wartosc_pln_akcje"] = merged["wartosc_pln_akcje"].fillna(0)
    merged["udzial_akcje_pct"] = merged.apply(
        lambda row: (row["wartosc_pln_akcje"] / row["wartosc_pln_total"] * 100)
        if row["wartosc_pln_total"] not in (0, None) and not pd.isna(row["wartosc_pln_total"])
        else None,
        axis=1,
    )
    merged = merged.sort_values(group_cols).reset_index(drop=True)
    return merged


def _format_percent(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        number = Decimal(str(value)) / Decimal("100")
    except (TypeError, ValueError, ArithmeticError):
        return ""
    text = format(number, "f").rstrip("0").rstrip(".")
    return text if text else "0"


def _most_common_emitent(series: pd.Series) -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned.str.lower().ne("nan") & cleaned.ne("")]
    if cleaned.empty:
        return ""
    return cleaned.value_counts().idxmax()


def build_equity_share_pivot(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"instytucja", "equity_nazwa", "TYP_aktywo_std", "wartosc_pln", "emitent"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    mask = _valid_group_mask(df, ["instytucja", "equity_nazwa"])
    work_df = df.loc[mask, ["instytucja", "equity_nazwa", "TYP_aktywo_std", "wartosc_pln", "emitent"]].copy()
    work_df = work_df[work_df["TYP_aktywo_std"].astype(str).str.strip().str.lower().eq("akcje")]
    if work_df.empty:
        return pd.DataFrame()

    work_df["wartosc_pln_num"] = work_df["wartosc_pln"].apply(parse_polish_number)
    work_df = work_df[work_df["wartosc_pln_num"].notna()]
    if work_df.empty:
        return pd.DataFrame()

    total_inst = (
        work_df.groupby("instytucja", dropna=False)["wartosc_pln_num"]
        .sum(min_count=1)
        .rename("wartosc_pln_inst")
    )
    equity_inst = (
        work_df.groupby(["equity_nazwa", "instytucja"], dropna=False)["wartosc_pln_num"]
        .sum(min_count=1)
        .reset_index()
    )
    equity_inst["udzial_pct"] = equity_inst.apply(
        lambda row: (row["wartosc_pln_num"] / total_inst.get(row["instytucja"], 0) * 100)
        if total_inst.get(row["instytucja"], 0) not in (0, None)
        else None,
        axis=1,
    )

    pivot = equity_inst.pivot_table(
        index="equity_nazwa",
        columns="instytucja",
        values="udzial_pct",
        aggfunc="sum",
    )
    pivot = pivot.reset_index().rename(columns={"equity_nazwa": "TICKER"})

    company_map = work_df.groupby("equity_nazwa")["emitent"].apply(_most_common_emitent)
    pivot.insert(0, "COMPANY", pivot["TICKER"].map(company_map).fillna(""))

    inst_cols = [c for c in pivot.columns if c not in ("COMPANY", "TICKER")]
    if inst_cols:
        pivot["TOTAL"] = pivot[inst_cols].sum(axis=1, skipna=True)
        ordered_cols = ["COMPANY", "TICKER"] + inst_cols + ["TOTAL"]
        pivot = pivot[ordered_cols]

    for col in pivot.columns:
        if col in ("COMPANY", "TICKER"):
            continue
        pivot[col] = pivot[col].apply(_format_percent)

    return pivot


def _format_change_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if str(value).strip().lower() in {"", "nan", "<na>"}:
        return ""
    try:
        number = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    except (TypeError, ValueError, ArithmeticError):
        return ""
    text = format(number, "f").rstrip("0").rstrip(".")
    return text


def _format_change_percent(value: Optional[float]) -> str:
    return _format_percent(value)


def _replace_nan_with_zero(value) -> str:
    if value is None:
        return "0"
    try:
        if pd.isna(value):
            return "0"
    except TypeError:
        pass
    if isinstance(value, str) and value.strip().lower() == "nan":
        return "0"
    return value


def build_equity_holdings_numeric(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"instytucja", "equity_nazwa", "TYP_aktywo_std", "liczba_sztuk", "emitent"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    mask = _valid_group_mask(df, ["instytucja", "equity_nazwa"])
    work_df = df.loc[mask, ["instytucja", "equity_nazwa", "TYP_aktywo_std", "liczba_sztuk", "emitent"]].copy()
    work_df = work_df[work_df["TYP_aktywo_std"].astype(str).str.strip().str.lower().eq("akcje")]
    if work_df.empty:
        return pd.DataFrame()

    work_df["liczba_sztuk_num"] = work_df["liczba_sztuk"].apply(parse_polish_number)
    work_df = work_df[work_df["liczba_sztuk_num"].notna()]
    if work_df.empty:
        return pd.DataFrame()

    holdings = (
        work_df.groupby(["equity_nazwa", "instytucja"], dropna=False)["liczba_sztuk_num"]
        .sum(min_count=1)
        .reset_index()
    )

    pivot = holdings.pivot_table(
        index="equity_nazwa",
        columns="instytucja",
        values="liczba_sztuk_num",
        aggfunc="sum",
    )
    pivot = pivot.reset_index().rename(columns={"equity_nazwa": "TICKER"})

    company_map = work_df.groupby("equity_nazwa")["emitent"].apply(_most_common_emitent)
    pivot.insert(0, "COMPANY", pivot["TICKER"].map(company_map).fillna(""))

    inst_cols = [c for c in pivot.columns if c not in ("COMPANY", "TICKER")]
    if inst_cols:
        pivot["TOTAL"] = pivot[inst_cols].sum(axis=1, skipna=True)
        ordered_cols = ["COMPANY", "TICKER"] + inst_cols + ["TOTAL"]
        pivot = pivot[ordered_cols]

    return pivot


def build_fund_position_share(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "instytucja",
        "fundusz",
        "equity_nazwa",
        "TYP_aktywo_std",
        "wartosc_pln",
        "liczba_sztuk",
        "emitent",
    }
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    mask = _valid_group_mask(df, ["instytucja", "fundusz", "equity_nazwa"])
    work_df = df.loc[
        mask,
        [
            "instytucja",
            "fundusz",
            "equity_nazwa",
            "TYP_aktywo_std",
            "wartosc_pln",
            "liczba_sztuk",
            "emitent",
        ],
    ].copy()
    work_df = work_df[work_df["TYP_aktywo_std"].astype(str).str.strip().str.lower().eq("akcje")]
    if work_df.empty:
        return pd.DataFrame()

    work_df["wartosc_pln_num"] = work_df["wartosc_pln"].apply(parse_polish_number)
    work_df["liczba_sztuk_num"] = work_df["liczba_sztuk"].apply(parse_polish_number)
    work_df = work_df[work_df["wartosc_pln_num"].notna()]
    if work_df.empty:
        return pd.DataFrame()

    fund_total = (
        work_df.groupby(["instytucja", "fundusz"], dropna=False)["wartosc_pln_num"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"wartosc_pln_num": "fund_total_pln_num"})
    )

    positions = (
        work_df.groupby(["instytucja", "fundusz", "equity_nazwa"], dropna=False)
        .agg(
            wartosc_pln_num=("wartosc_pln_num", "sum"),
            liczba_sztuk_num=("liczba_sztuk_num", "sum"),
        )
        .reset_index()
        .rename(columns={"equity_nazwa": "TICKER"})
    )

    positions = positions.merge(fund_total, on=["instytucja", "fundusz"], how="left")
    positions["fund_pct"] = positions.apply(
        lambda row: (row["wartosc_pln_num"] / row["fund_total_pln_num"] * 100)
        if row.get("fund_total_pln_num") not in (0, None) and not pd.isna(row.get("fund_total_pln_num"))
        else None,
        axis=1,
    )

    company_map = work_df.groupby("equity_nazwa")["emitent"].apply(_most_common_emitent)
    positions.insert(0, "COMPANY", positions["TICKER"].map(company_map).fillna(""))

    ordered_cols = [
        "instytucja",
        "fundusz",
        "COMPANY",
        "TICKER",
        "liczba_sztuk_num",
        "wartosc_pln_num",
        "fund_total_pln_num",
        "fund_pct",
    ]
    return positions[ordered_cols]


def build_change_table(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    if prev_df.empty and curr_df.empty:
        return pd.DataFrame()

    prev = prev_df.copy()
    curr = curr_df.copy()

    prev_inst_cols = [c for c in prev.columns if c not in ("COMPANY", "TICKER")]
    curr_inst_cols = [c for c in curr.columns if c not in ("COMPANY", "TICKER")]
    ordered = []
    for col in curr_inst_cols + prev_inst_cols:
        if col not in ordered and col != "TOTAL":
            ordered.append(col)
    if "TOTAL" in curr_inst_cols or "TOTAL" in prev_inst_cols:
        ordered.append("TOTAL")
    inst_cols = ordered

    prev_map = prev.set_index("TICKER") if "TICKER" in prev.columns else pd.DataFrame()
    curr_map = curr.set_index("TICKER") if "TICKER" in curr.columns else pd.DataFrame()
    tickers = sorted(set(prev_map.index.tolist()) | set(curr_map.index.tolist()))

    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        prev_row = prev_map.loc[ticker] if ticker in prev_map.index else pd.Series()
        curr_row = curr_map.loc[ticker] if ticker in curr_map.index else pd.Series()
        company = ""
        if "COMPANY" in curr_row and pd.notna(curr_row.get("COMPANY")):
            company = str(curr_row.get("COMPANY"))
        elif "COMPANY" in prev_row and pd.notna(prev_row.get("COMPANY")):
            company = str(prev_row.get("COMPANY"))

        row: Dict[str, object] = {"COMPANY": company, "TICKER": ticker}
        for col in inst_cols:
            prev_val = prev_row.get(col, 0) if isinstance(prev_row, pd.Series) else 0
            curr_val = curr_row.get(col, 0) if isinstance(curr_row, pd.Series) else 0
            prev_val = 0 if pd.isna(prev_val) else prev_val
            curr_val = 0 if pd.isna(curr_val) else curr_val
            delta = curr_val - prev_val
            row[col] = delta
            if prev_val == 0:
                row[f"{col}_pct"] = None
            else:
                row[f"{col}_pct"] = ((curr_val / prev_val) - 1) * 100
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    base_cols = ["COMPANY", "TICKER"] + inst_cols
    pct_cols = [f"{col}_pct" for col in inst_cols]
    result = result[base_cols + pct_cols]
    result = result.sort_values(["COMPANY", "TICKER"], kind="mergesort").reset_index(drop=True)

    for col in inst_cols:
        result[col] = result[col].apply(_format_change_number)
    for col in pct_cols:
        result[col] = result[col].apply(_format_change_percent)

    return result





def build_fund_position_changes(
    fund_positions_by_quarter: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not fund_positions_by_quarter:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for quarter_token in sorted(fund_positions_by_quarter.keys(), key=quarter_sort_key):
        prev_token = prev_quarter_token(quarter_token)
        if prev_token not in fund_positions_by_quarter:
            continue

        prev_df = fund_positions_by_quarter.get(prev_token, pd.DataFrame())
        curr_df = fund_positions_by_quarter.get(quarter_token, pd.DataFrame())
        if prev_df.empty and curr_df.empty:
            continue

        merged = curr_df.merge(
            prev_df,
            on=["instytucja", "fundusz", "TICKER"],
            how="outer",
            suffixes=("_curr", "_prev"),
        )

        if merged.empty:
            continue

        merged["COMPANY"] = merged.get("COMPANY_curr").combine_first(
            merged.get("COMPANY_prev")
        )
        merged["wartosc_pln_chg"] = (
            merged.get("wartosc_pln_num_curr", 0).fillna(0)
            - merged.get("wartosc_pln_num_prev", 0).fillna(0)
        )
        merged["liczba_sztuk_chg"] = (
            merged.get("liczba_sztuk_num_curr", 0).fillna(0)
            - merged.get("liczba_sztuk_num_prev", 0).fillna(0)
        )

        prev_val = merged.get("wartosc_pln_num_prev", 0).fillna(0)
        merged["wartosc_pln_chg_pct"] = prev_val.where(prev_val != 0)
        merged["wartosc_pln_chg_pct"] = (
            (merged.get("wartosc_pln_num_curr", 0).fillna(0) / merged["wartosc_pln_chg_pct"]) - 1
        ) * 100

        prev_qty = merged.get("liczba_sztuk_num_prev", 0).fillna(0)
        merged["liczba_sztuk_chg_pct"] = prev_qty.where(prev_qty != 0)
        merged["liczba_sztuk_chg_pct"] = (
            (merged.get("liczba_sztuk_num_curr", 0).fillna(0) / merged["liczba_sztuk_chg_pct"]) - 1
        ) * 100

        merged.insert(0, "quarter", quarter_token)
        merged.insert(1, "yoy_prev", prev_token)

        frames.append(
            merged[
                [
                    "quarter",
                    "yoy_prev",
                    "instytucja",
                    "fundusz",
                    "COMPANY",
                    "TICKER",
                    "wartosc_pln_chg",
                    "wartosc_pln_chg_pct",
                    "liczba_sztuk_chg",
                    "liczba_sztuk_chg_pct",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _normalize_master_match_values(
    quarter_token: str,
    instytucja: str,
    fundusz: str,
) -> Tuple[str, str]:
    inst = str(instytucja or "").strip()
    fund = str(fundusz or "").strip()
    if quarter_token not in {"2Q25", "2Q26"}:
        return inst, fund

    em_match = re.match(
        r"^(?:Goldman Sachs|ING)\s+Emerytura\s+(\d{4})$",
        fund,
        re.IGNORECASE,
    )
    if em_match:
        return "EMERYTURA_ING_GOLDMAN", f"Emerytura {em_match.group(1)}"

    ppk_match = re.match(
        r"^(?:Santander|Erste)\s+PPK\s+(\d{4})$",
        fund,
        re.IGNORECASE,
    )
    if ppk_match:
        return "PPK_SANTANDER_ERSTE", f"PPK {ppk_match.group(1)}"

    return inst, fund


def build_master_dataset(
    source_rows_by_quarter: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not source_rows_by_quarter:
        return pd.DataFrame()

    match_cols = [
        "instytucja",
        "fundusz",
        "DATA_fundusz",
        "emitent",
        "isin",
        "waluta",
        "TYP_aktywo_std",
        "equity_nazwa",
    ]

    quarter_frames: List[pd.DataFrame] = []
    normalized_by_quarter: Dict[str, pd.DataFrame] = {}

    for quarter_token, df in source_rows_by_quarter.items():
        if df.empty:
            continue

        work = df.copy()
        for col in match_cols + ["typ_aktywa", "data", "liczba_sztuk", "wartosc_pln"]:
            if col not in work.columns:
                work[col] = ""

        work["liczba_sztuk_num"] = work["liczba_sztuk"].apply(parse_polish_number)
        work["wartosc_pln_num"] = work["wartosc_pln"].apply(parse_polish_number)

        fund_total = (
            work.groupby(["instytucja", "fundusz"], dropna=False)["wartosc_pln_num"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"wartosc_pln_num": "fund_total_pln_num"})
        )
        work = work.merge(fund_total, on=["instytucja", "fundusz"], how="left")
        work["fund_pct_num"] = work.apply(
            lambda row: (row["wartosc_pln_num"] / row["fund_total_pln_num"] * 100)
            if row.get("wartosc_pln_num") not in (None,) and not pd.isna(row.get("wartosc_pln_num"))
            and row.get("fund_total_pln_num") not in (0, None) and not pd.isna(row.get("fund_total_pln_num"))
            else None,
            axis=1,
        )

        work["quarter"] = quarter_token
        work.insert(0, "quarter", work.pop("quarter"))
        inst = work["instytucja"].astype(str).str.strip()
        fundusz = work["fundusz"].astype(str).str.strip()
        data_fund = work["DATA_fundusz"].astype(str).str.strip()
        isin = work["isin"].astype(str).str.strip()
        emitent = work["emitent"].astype(str).str.strip()
        waluta = work["waluta"].astype(str).str.strip()
        typ_std = work["TYP_aktywo_std"].astype(str).str.strip()

        normalized = [
            _normalize_master_match_values(q, i, f)
            for q, i, f in zip(
                [quarter_token] * len(work),
                inst.tolist(),
                fundusz.tolist(),
            )
        ]
        normalized_inst, normalized_fund = zip(*normalized)
        normalized_inst = pd.Series(normalized_inst, index=work.index)
        normalized_fund = pd.Series(normalized_fund, index=work.index)

        isin_mask = isin.str.lower().ne("nan") & isin.ne("")
        key_isin = normalized_inst + "|" + normalized_fund + "|" + data_fund + "|" + isin
        key_other = (
            normalized_inst
            + "|"
            + normalized_fund
            + "|"
            + data_fund
            + "|"
            + emitent
            + "|"
            + waluta
            + "|"
            + typ_std
        )

        work["match_key"] = key_other
        work.loc[isin_mask, "match_key"] = key_isin
        work["_dup_idx"] = work.groupby(["match_key"], dropna=False).cumcount()

        normalized_by_quarter[quarter_token] = work.copy()
        quarter_frames.append(work)

    if not quarter_frames:
        return pd.DataFrame()

    master = pd.concat(quarter_frames, ignore_index=True)
    master["liczba_sztuk_chg_num"] = pd.NA
    master["liczba_sztuk_chg_pct_num"] = pd.NA
    master["wartosc_pln_chg_num"] = pd.NA
    master["wartosc_pln_chg_pct_num"] = pd.NA

    for quarter_token in sorted(normalized_by_quarter.keys(), key=quarter_sort_key):
        prev_token = prev_quarter_token(quarter_token)
        if prev_token not in normalized_by_quarter:
            continue

        curr = normalized_by_quarter[quarter_token].copy()
        prev = normalized_by_quarter[prev_token].copy()

        curr_idx = curr.index.to_series().rename("_local_idx")
        curr = curr.reset_index(drop=True)
        curr["_local_idx"] = curr_idx.values

        merged = curr.merge(
            prev[
                ["match_key"]
                + [
                    "_dup_idx",
                    "liczba_sztuk_num",
                    "wartosc_pln_num",
                ]
            ],
            on=["match_key", "_dup_idx"],
            how="left",
            suffixes=("_curr", "_prev"),
        )

        qty_curr = merged.get("liczba_sztuk_num_curr")
        qty_prev = merged.get("liczba_sztuk_num_prev")
        val_curr = merged.get("wartosc_pln_num_curr")
        val_prev = merged.get("wartosc_pln_num_prev")

        qty_chg = qty_curr.fillna(0) - qty_prev.fillna(0)
        val_chg = val_curr.fillna(0) - val_prev.fillna(0)

        qty_chg_pct = ((qty_curr / qty_prev) - 1) * 100
        val_chg_pct = ((val_curr / val_prev) - 1) * 100

        qty_chg_pct = qty_chg_pct.where(qty_prev.notna() & qty_prev.ne(0))
        val_chg_pct = val_chg_pct.where(val_prev.notna() & val_prev.ne(0))

        target_mask = master["quarter"].astype(str).eq(quarter_token)
        target_indices = master[target_mask].index

        master.loc[target_indices, "liczba_sztuk_chg_num"] = qty_chg.values
        master.loc[target_indices, "wartosc_pln_chg_num"] = val_chg.values
        master.loc[target_indices, "liczba_sztuk_chg_pct_num"] = qty_chg_pct.values
        master.loc[target_indices, "wartosc_pln_chg_pct_num"] = val_chg_pct.values

    prev_qty_from_chg = master["liczba_sztuk_num"] - master["liczba_sztuk_chg_num"]
    prev_val_from_chg = master["wartosc_pln_num"] - master["wartosc_pln_chg_num"]

    missing_qty_pct = master["liczba_sztuk_chg_pct_num"].isna()
    missing_val_pct = master["wartosc_pln_chg_pct_num"].isna()

    qty_mask = missing_qty_pct & prev_qty_from_chg.notna() & prev_qty_from_chg.ne(0)
    val_mask = missing_val_pct & prev_val_from_chg.notna() & prev_val_from_chg.ne(0)

    master.loc[qty_mask, "liczba_sztuk_chg_pct_num"] = (
        (master.loc[qty_mask, "liczba_sztuk_num"] / prev_qty_from_chg[qty_mask]) - 1
    ) * 100
    master.loc[val_mask, "wartosc_pln_chg_pct_num"] = (
        (master.loc[val_mask, "wartosc_pln_num"] / prev_val_from_chg[val_mask]) - 1
    ) * 100

    proxy_qty_mask = missing_qty_pct & prev_qty_from_chg.eq(0) & master["liczba_sztuk_num"].notna() & master["liczba_sztuk_num"].ne(0)
    proxy_val_mask = missing_val_pct & prev_val_from_chg.eq(0) & master["wartosc_pln_num"].notna() & master["wartosc_pln_num"].ne(0)

    master.loc[proxy_qty_mask, "liczba_sztuk_chg_pct_num"] = (
        master.loc[proxy_qty_mask, "liczba_sztuk_chg_num"] / master.loc[proxy_qty_mask, "liczba_sztuk_num"]
    ) * 100
    master.loc[proxy_val_mask, "wartosc_pln_chg_pct_num"] = (
        master.loc[proxy_val_mask, "wartosc_pln_chg_num"] / master.loc[proxy_val_mask, "wartosc_pln_num"]
    ) * 100

    master["fund_total_pln"] = master["fund_total_pln_num"].apply(format_decimal_comma)
    master["fund_pct"] = master["fund_pct_num"].apply(_format_percent)
    master["liczba_sztuk_chg"] = master["liczba_sztuk_chg_num"].apply(format_decimal_comma)
    master["wartosc_pln_chg"] = master["wartosc_pln_chg_num"].apply(format_decimal_comma)
    master["liczba_sztuk_chg_pct"] = master["liczba_sztuk_chg_pct_num"].apply(_format_change_percent)
    master["wartosc_pln_chg_pct"] = master["wartosc_pln_chg_pct_num"].apply(_format_change_percent)
    master["liczba_sztuk_chg"] = master["liczba_sztuk_chg"].replace("nan", "")
    master["wartosc_pln_chg"] = master["wartosc_pln_chg"].replace("nan", "")

    master["Institutions_actual"] = master["instytucja"].astype(str).str.strip()
    master.loc[
        master["instytucja"].astype(str).str.strip().eq("Santander TFI S.A."),
        "Institutions_actual",
    ] = "Erste"
    master.loc[
        master["instytucja"].astype(str).str.strip().eq("Goldman Sachs TFI S.A."),
        "Institutions_actual",
    ] = "ING"

    ordered_cols = [
        "quarter",
        "data",
        "instytucja",
        "Institutions_actual",
        "fundusz",
        "DATA_fundusz",
        "typ_aktywa",
        "emitent",
        "isin",
        "waluta",
        "liczba_sztuk",
        "wartosc_pln",
        "TYP_aktywo_std",
        "equity_nazwa",
        "fund_total_pln",
        "fund_pct",
        "liczba_sztuk_chg",
        "liczba_sztuk_chg_pct",
        "wartosc_pln_chg",
        "wartosc_pln_chg_pct",
    ]

    for col in ordered_cols:
        if col not in master.columns:
            master[col] = ""

    master = master.reindex(columns=ordered_cols)
    master = master.apply(lambda col: col.map(_replace_nan_with_zero))
    return master


def cleanup_percent_outputs(output_dir: str) -> None:
    patterns = ["*_akcje_*.csv", "*_fund_share.csv", "*_holdings_pct.csv", "*_chg.csv"]
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(path)
            except OSError:
                continue


def extract_tables_from_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
    table_settings: Optional[dict] = None,
) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            page_tables = []
            try:
                if table_settings is not None:
                    page_tables = page.extract_tables(table_settings)
                else:
                    page_tables = page.extract_tables()
            except Exception:
                page_tables = []
            if not page_tables:
                try:
                    page_tables = page.extract_tables(PDF_TABLE_SETTINGS)
                except Exception:
                    page_tables = []
            for table in page_tables or []:
                if not table or len(table) < 2:
                    continue
                headers = [normalize_header(h) for h in table[0]]
                headers = make_unique_headers(headers)
                rows = table[1:]
                df = pd.DataFrame(rows, columns=headers)
                df = df.dropna(axis=1, how="all")
                df = df.loc[:, ~df.columns.duplicated()]
                tables.append(df)
    return tables


def detect_allianz_header_row(df_raw: pd.DataFrame, max_rows: int = 50) -> int:
    for idx in range(min(max_rows, len(df_raw))):
        row = df_raw.iloc[idx]
        joined = " ".join([str(x) for x in row.tolist() if pd.notna(x)])
        if re.search(r"nazwa funduszu", joined, re.IGNORECASE):
            return idx
        if re.search(r"kod isin", joined, re.IGNORECASE):
            return idx
    return 0


def select_pzu_excel_sheet(file_path: str, expected_keywords: Optional[List[str]] = None) -> Optional[str]:
    try:
        xls = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception:
        return None

    sheet_names = xls.sheet_names or []
    if not sheet_names:
        return None

    if expected_keywords:
        best_sheet: Optional[str] = None
        best_score = -1
        for sheet_name in sheet_names:
            try:
                df_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=None, nrows=20)
            except Exception:
                continue

            text = " ".join([str(value) for value in df_raw.stack().tolist() if pd.notna(value)]).lower()
            score = sum(1 for keyword in expected_keywords if keyword.lower() in text)
            if score > best_score:
                best_sheet = sheet_name
                best_score = score

        if best_sheet is not None and best_score > 0:
            return best_sheet

    return sheet_names[0]


def extract_date_from_excel(df_raw: pd.DataFrame) -> str:
    for value in df_raw.stack().tolist():
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            value = value.strip()
            if re.match(r"\d{4}-\d{2}-\d{2}", value):
                return value
            if re.match(r"\d{2}\.\d{2}\.\d{4}", value):
                try:
                    return datetime.strptime(value, "%d.%m.%Y").strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return ""


def extract_date_from_santander(df_raw: pd.DataFrame) -> str:
    for row_idx in range(min(20, len(df_raw))):
        row = df_raw.iloc[row_idx].tolist()
        for col_idx, value in enumerate(row):
            if isinstance(value, str) and "skład portfeli na dzień" in value.lower():
                for next_idx in range(col_idx + 1, len(row)):
                    candidate = row[next_idx]
                    if isinstance(candidate, datetime):
                        return candidate.strftime("%Y-%m-%d")
                    if isinstance(candidate, str):
                        candidate = candidate.strip()
                        if re.match(r"\d{2}\.\d{2}\.\d{4}", candidate):
                            try:
                                return datetime.strptime(candidate, "%d.%m.%Y").strftime("%Y-%m-%d")
                            except ValueError:
                                continue
                        if re.match(r"\d{4}-\d{2}-\d{2}", candidate):
                            return candidate
    return ""


def extract_date_from_filename(file_path: str) -> str:
    name = os.path.basename(file_path)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d.%m.%Y").strftime("%Y-%m-%d")
        except ValueError:
            return ""
    return ""


def is_millennium_excel(file_path: str) -> bool:
    try:
        df_raw = pd.read_excel(file_path, engine="openpyxl", header=None, nrows=15)
    except Exception:
        return False
    for i in range(len(df_raw)):
        row_text = " ".join(
            [str(x) for x in df_raw.iloc[i].tolist() if pd.notna(x)]
        ).lower()
        if "nazwa subfunduszu" in row_text and "wartość instrumentu w pln" in row_text:
            return True
    return False


def select_millennium_sheet(file_path: str) -> Optional[str]:
    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception:
        return None

    file_date = extract_date_from_filename(file_path)
    if file_date:
        try:
            dt = datetime.strptime(file_date, "%Y-%m-%d")
            token = dt.strftime("%d.%m.%Y")
        except ValueError:
            token = ""
        if token:
            for sheet in xl.sheet_names:
                sheet_norm = normalize_header(sheet)
                if "skład portfela" not in sheet_norm and "skład porfela" not in sheet_norm:
                    continue
                match = re.search(r"(\d{2}\.\d{2}\.\d{4})", sheet_norm)
                if match and match.group(1) == token:
                    return sheet
    return xl.sheet_names[-1] if xl.sheet_names else None


def extract_text_from_pdf(pdf_path: str, max_pages: int = 2) -> str:
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_simple_from_pdf(pdf_path: str, max_pages: int = 2) -> str:
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            extract_simple = getattr(page, "extract_text_simple", None)
            if not callable(extract_simple):
                texts.append("")
                continue

            def _timeout_handler(signum, frame):
                raise TimeoutError()

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            try:
                signal.alarm(2)
                texts.append(extract_simple() or "")
            except TimeoutError:
                texts.append("")
            except Exception:
                texts.append("")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    return "\n".join(texts)


def extract_text_pdfminer(
    pdf_path: str,
    page_numbers: Optional[List[int]] = None,
    timeout_seconds: int = 4,
) -> str:
    if pdfminer_extract_text is None:
        return ""

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(timeout_seconds)
        return pdfminer_extract_text(pdf_path, page_numbers=page_numbers) or ""
    except TimeoutError:
        return ""
    except Exception:
        return ""
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_text_pdfplumber_simple(pdf_path: str, max_pages: Optional[int] = None) -> str:
    texts: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                try:
                    texts.append(page.extract_text_simple() or "")
                except Exception:
                    texts.append("")
    except Exception:
        return ""
    return "\f".join(texts)


def extract_date_from_title(file_path: str) -> str:
    file_date = extract_date_from_filename(file_path)
    if file_date:
        return file_date

    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        preview_text = extract_text_pdfminer(file_path, page_numbers=[0], timeout_seconds=6)
        if not preview_text:
            preview_text = extract_text_from_pdf(file_path, max_pages=1)
        if not preview_text:
            preview_text = extract_text_simple_from_pdf(file_path, max_pages=1)

        parsed = parse_date_from_text(preview_text or "")
        if parsed:
            return parsed

        match = re.search(r"(\d{1,2})[./\s](\d{1,2})[./\s](\d{4})", preview_text or "")
        if match:
            try:
                day, month, year = match.groups()
                return datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                return ""

        return ""

    df_raw = None
    try:
        if lower_path.endswith((".xls", ".xlsx")):
            df_raw = pd.read_excel(file_path, engine="openpyxl", header=None, nrows=15)
        elif lower_path.endswith(".csv"):
            df_raw = pd.read_csv(file_path, header=None, nrows=15, sep=None, engine="python")
    except Exception:
        df_raw = None

    if df_raw is not None:
        return extract_date_from_excel(df_raw)

    return ""


def fill_missing_date_from_title(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    if df.empty or "data" not in df.columns:
        return df

    data_series = df["data"].astype(str).str.strip()
    missing_mask = data_series.eq("") | data_series.str.lower().eq("nan")
    if not missing_mask.any():
        return df

    title_date = extract_date_from_title(file_path)
    if not title_date:
        return df

    df.loc[missing_mask, "data"] = title_date
    return df


def parse_date_from_text(text: str) -> Optional[str]:
    patterns = [
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{2}\.\d{2}\.\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1)
            try:
                if "." in raw:
                    dt = datetime.strptime(raw, "%d.%m.%Y")
                else:
                    dt = datetime.strptime(raw, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None





# -------------------------
# Parsers
# -------------------------

def parse_allianz_excel(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
    header_row = detect_allianz_header_row(df_raw)
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
    file_name = os.path.basename(file_path)
    file_date = extract_date_from_filename(file_path)

    mapping = {
        "data sporządzenia": "data",
        "nazwa funduszu/subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "emitent": "emitent",
        "kod isin instrumentu": "isin",
        "waluta wyceny instrumentu": "waluta",
        "ilość": "liczba_sztuk",
        "wartość bilansowa": "wartosc_pln",
    }

    df.columns = [normalize_header(c) for c in df.columns]
    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "Allianz Polska TFI S.A."
    df["fundusz"] = df.get("fundusz", "")
    df["data"] = file_date if file_date else df.get("data", "")
    df = df.dropna(axis=0, how="all")

    out = ensure_output_schema(df)

    if file_name == "pzu1_2023-12-31.xlsx" and not out.empty:
        target_years = set(_PZU_EOP_2023_VALUES.keys())
        year_norm = (
            out["DATA_fundusz"]
            .astype(str)
            .str.extract(r"(20\d{2})", expand=False)
            .fillna("")
        )
        pzu_mask = (
            year_norm.isin(target_years)
            & (
                out.get("isin", "").astype(str).str.upper().str.strip().eq("PLPZU0000011")
                | out.get("emitent", "").astype(str).str.contains(r"\bPZU\b", case=False, regex=True, na=False)
            )
        )

        template = {col: "" for col in out.columns}
        sample = out.iloc[0].to_dict()
        template.update({k: ("" if pd.isna(v) else v) for k, v in sample.items()})

        out = out.loc[~pzu_mask].copy()
        rows: List[Dict[str, Any]] = []
        for year, corrected_value in _PZU_EOP_2023_VALUES.items():
            row = dict(template)
            row["fundusz"] = f"inPZU Puls Życia {year}"
            row["DATA_fundusz"] = year
            row["typ_aktywa"] = "Akcje"
            row["emitent"] = "PZU"
            row["isin"] = "PLPZU0000011"
            row["waluta"] = "PLN"
            row["liczba_sztuk"] = ""
            row["wartosc_pln"] = corrected_value
            row["TYP_aktywo_std"] = "akcje"
            rows.append(row)
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

        # Remove known legacy duplicates coming from raw source rows.
        value_num = out["wartosc_pln"].apply(parse_polish_number)
        year_norm_final = (
            out["DATA_fundusz"]
            .astype(str)
            .str.extract(r"(20\d{2})", expand=False)
            .fillna("")
        )
        legacy_wrong = {
            "2030": 4_184_000,
            "2040": 7_970_000,
            "2060": 339_000,
        }
        legacy_mask = pd.Series(False, index=out.index)
        for year, wrong_value in legacy_wrong.items():
            legacy_mask = legacy_mask | (
                year_norm_final.eq(year)
                & out.get("isin", "").astype(str).str.upper().str.strip().eq("PLPZU0000011")
                & value_num.eq(float(wrong_value))
            )
        if legacy_mask.any():
            out = out.loc[~legacy_mask].copy()

    return ensure_output_schema(out)


def parse_santander_excel(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
    extracted_date = extract_date_from_santander(df_raw)
    header_row = detect_allianz_header_row(df_raw)
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)

    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta": "emitent",
        "identyfikator instrumentu - kod isin": "isin",
        "waluta wykorzystywana do wyceny instrumentu": "waluta",
        "ilość instrumentu w portfelu": "liczba_sztuk",
        "wartość instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "Santander TFI S.A."
    if extracted_date:
        df["data"] = df.get("data", "")
        df.loc[df["data"].astype(str).str.strip() == "", "data"] = extracted_date

    fundusz_col = df.get("fundusz", "").astype(str)
    df = df[fundusz_col.str.contains("PPK", case=False, na=False)]
    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_erste_excel(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
    header_row = detect_allianz_header_row(df_raw)
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta": "emitent",
        "identyfikator instrumentu - kod isin": "isin",
        "waluta wyceny aktywów i zobowiązań funduszu": "waluta",
        "ilość instrumentu w portfelu": "liczba_sztuk",
        "wartość instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "Erste"
    file_date = extract_date_from_filename(file_path)
    df["data"] = file_date if file_date else ""

    fundusz_series = df.get("fundusz", "").astype(str)
    ppk_mask = fundusz_series.str.contains("PPK", case=False, na=False)
    df = df[ppk_mask].copy()

    if df.empty:
        print(f"Erste parser: brak rekordów PPK w pliku {os.path.basename(file_path)}")
        return ensure_output_schema(df)

    def _extract_data_fundusz(value: str) -> str:
        match = re.search(r"(20\d{2})", str(value))
        return match.group(1) if match else ""

    df["DATA_fundusz"] = df["fundusz"].astype(str).apply(_extract_data_fundusz)
    missing_year_mask = df["DATA_fundusz"].astype(str).str.strip() == ""
    if missing_year_mask.any():
        count_missing = int(missing_year_mask.sum())
        print(
            f"Erste parser: nie udało się wyodrębnić DATA_fundusz dla {count_missing} rekordów w pliku {os.path.basename(file_path)}"
        )

    missing_isin_mask = df.get("isin", "").astype(str).str.strip().eq("")
    if missing_isin_mask.any():
        count_missing_isin = int(missing_isin_mask.sum())
        print(
            f"Erste parser: {count_missing_isin} rekordów bez ISIN pozostało w pliku {os.path.basename(file_path)}"
        )

    total_rows = len(df)
    print(f"Erste parser: zaimportowano {total_rows} rekordów z pliku {os.path.basename(file_path)}")

    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_ing_excel(file_path: str) -> pd.DataFrame:
    file_date = extract_date_from_filename(file_path)

    try:
        df = pd.read_excel(file_path, engine="openpyxl", header=0)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa funduszu / nazwa subfunduszu": "fundusz",
        "kategoria / typ instrumentu": "typ_aktywa",
        "isin": "isin",
        "nazwa pełna instrumentu": "emitent",
        "nazwa emitenta / nazwa wystawcy instrumentu pochodnego otc": "emitent",
        "waluta": "waluta",
        "ilość": "liczba_sztuk",
        "wartość całkowita": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "ING"
    df["data"] = file_date

    fundusz_series = df.get("fundusz", "").astype(str)
    df["fundusz"] = fundusz_series

    def _extract_data_fundusz(value: str) -> str:
        match = re.search(r"(20\d{2})", str(value))
        return match.group(1) if match else ""

    df["DATA_fundusz"] = df["fundusz"].astype(str).apply(_extract_data_fundusz)
    missing_year_mask = df["DATA_fundusz"].astype(str).str.strip() == ""
    if missing_year_mask.any() and file_date:
        df.loc[missing_year_mask, "DATA_fundusz"] = file_date[:4]

    keep_mask = df["fundusz"].astype(str).str.strip().str.match(
        r"^(?:ing|goldman sachs) emerytura 20\d{2}$",
        case=False,
        na=False,
    )
    dropped = int((~keep_mask).sum())
    if dropped:
        print(
            f"ING parser: odrzucono {dropped} wierszy, ponieważ nie należą do funduszy ING Emerytura 20xx lub Goldman Sachs Emerytura 20xx"
        )
    df = df.loc[keep_mask].copy()

    if not file_date:
        print(f"ING parser: nie udało się wyodrębnić daty z nazwy pliku {os.path.basename(file_path)}")

    missing_isin_mask = df.get("isin", "").astype(str).str.strip().eq("")
    if missing_isin_mask.any():
        count_missing_isin = int(missing_isin_mask.sum())
        print(
            f"ING parser: {count_missing_isin} rekordów bez ISIN pozostało w pliku {os.path.basename(file_path)}"
        )

    total_rows = len(df)
    print(f"ING parser: zaimportowano {total_rows} rekordów z pliku {os.path.basename(file_path)}")

    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_bnp_excel(file_path: str) -> pd.DataFrame:
    # Ekstrakcja daty z nazwy pliku (BNP_YYYY-MM-DD.xlsx)
    file_date = extract_date_from_filename(file_path)
    
    # Wczytanie danych z nagłówkiem w rzędzie 0
    df = pd.read_excel(file_path, engine="openpyxl", header=0)
    
    # Normalizacja nagłówków
    df.columns = [normalize_header(c) for c in df.columns]
    
    # Mapowanie kolumn
    mapping = {
        "nazwa subfunduszu": "fundusz",
        "identyfikator instrumentu - kod isin": "isin",
        "ilość instrumentu w portfelu": "liczba_sztuk",
        "wartość instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        if dst in ["liczba_sztuk", "wartosc_pln"]:
            df[dst] = df[col].apply(parse_polish_number) if col else None
        else:
            df[dst] = df[col] if col else ""

    typ_col = find_column(df, "typ instrumentu") or find_column(df, "rodzaj instrumentu")
    emitent_col = find_column(df, "nazwa emitenta") or find_column(df, "emitent")
    waluta_col = (
        find_column(df, "waluta wykorzystywana do wyceny instrumentu")
        or find_column(df, "waluta wyceny aktywów i zobowiązań funduszu")
        or find_column(df, "waluta")
    )
    df["typ_aktywa"] = df[typ_col] if typ_col else ""
    df["emitent"] = df[emitent_col] if emitent_col else ""
    df["waluta"] = df[waluta_col] if waluta_col else ""
    
    # Ustawienie instytucji i daty
    df["instytucja"] = "BNP Paribas TFI S.A."
    if file_date:
        df["data"] = file_date
    else:
        df["data"] = ""
    
    # Filtrowanie tylko wierszy zawierających "PPK" w nazwie subfunduszu
    fundusz_col = df.get("fundusz", "").astype(str)
    df = df[fundusz_col.str.contains("PPK", case=False, na=False)]
    
    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_goldman_excel(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
    header_row = 0
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa funduszu / nazwa subfunduszu": "fundusz",
        "kategoria / typ instrumentu": "typ_aktywa",
        "nazwa emitenta / nazwa wystawcy instrumentu pochodnego otc": "emitent",
        "isin": "isin",
        "waluta": "waluta",
        "ilość": "liczba_sztuk",
        "wartość całkowita": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "Goldman Sachs TFI S.A."
    file_date = extract_date_from_filename(file_path)
    if file_date:
        df["data"] = df.get("data", "")
        df.loc[df["data"].astype(str).str.strip() == "", "data"] = file_date

    fundusz_col = df.get("fundusz", "").astype(str)
    df = df[fundusz_col.str.contains("Emerytura", case=False, na=False)]
    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_millennium_excel(file_path: str) -> pd.DataFrame:
    sheet_name = select_millennium_sheet(file_path)
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None, sheet_name=sheet_name)
    header_row = 7
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row, sheet_name=sheet_name)
    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta": "emitent",
        "identyfikator instrumentu (isin)": "isin",
        "waluta instrumentu": "waluta",
        "ilość instrumentów w portfelu": "liczba_sztuk",
        "wartość instrumentu w pln": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "Millennium TFI S.A."
    file_date = extract_date_from_filename(file_path)
    if file_date:
        df["data"] = df.get("data", "")
        df.loc[df["data"].astype(str).str.strip() == "", "data"] = file_date

    fundusz_col = df.get("fundusz", "").astype(str)
    df = df[fundusz_col.str.contains("Emerytura", case=False, na=False)]
    df = df.dropna(axis=0, how="all")
    return ensure_output_schema(df)


def parse_pfr_excel(file_path: str) -> pd.DataFrame:
    """
    Parse PFR TFI Excel/CSV files.
    Assumes headers are in the first row.
    Maps PFR-specific columns to the standard OUTPUT_COLUMNS.
    """
    try:
        df = pd.read_excel(file_path, engine="openpyxl", header=0)
    except Exception:
        try:
            df = pd.read_csv(file_path, header=0)
        except Exception:
            return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    mapping_candidates = {
        "fundusz": ["nazwa subfunduszu"],
        "data": ["data wyceny"],
        "typ_aktywa": ["rodzaj instrumentu"],
        "emitent": ["emitent"],
        "isin": ["kod isin"],
        "waluta": ["waluta notowań", "waluta nominału", "waluta nominalu"],
        "liczba_sztuk": ["ilość", "ilosc"],
        "wartosc_pln": [
            "wartość całkowita w walucie wyceny funduszu",
            "wartosc całkowita w walucie wyceny funduszu",
            "wartość całkowita",
            "wartosc całkowita",
        ],
    }

    for dst, sources in mapping_candidates.items():
        selected_col = None
        for src in sources:
            selected_col = find_column(df, src)
            if selected_col:
                break
        df[dst] = df[selected_col] if selected_col else ""

    df["instytucja"] = "PFR TFI S.A."

    file_date = extract_date_from_filename(file_path)
    if file_date:
        df["data"] = df.get("data", "")
        df.loc[df["data"].astype(str).str.strip() == "", "data"] = file_date

    df = df.dropna(axis=0, how="all")

    return ensure_output_schema(df)


def parse_pko_excel(file_path: str, fallback_date: Optional[str] = None) -> pd.DataFrame:
    """
    Parse PKO TFI Excel files.
    Assumes headers are in the first row.
    Maps PKO-specific columns to the standard OUTPUT_COLUMNS.
    """
    try:
        df = pd.read_excel(file_path, engine="openpyxl", header=0)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta": "emitent",
        "identyfikator instrumentu": "isin",
        "waluta instrumentu": "waluta",
        "ilosc instrumentow w portfelu": "liczba_sztuk",
        "wartosc instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        if col:
            df[dst] = df[col]
        else:
            df[dst] = ""

    df["instytucja"] = "PKO TFI S.A."

    file_date = extract_date_from_filename(file_path)
    if not file_date:
        file_date = fallback_date or ""
    df["data"] = file_date

    # Filter out rows where fundusz or isin is empty (including "nan" string)
    def is_not_empty(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s != "" and s != "nan" and s != "none"

    df = df[df["fundusz"].apply(is_not_empty) | df["isin"].apply(is_not_empty)]

    # Remove rows that have no meaningful data in key fields
    df = df[
        df["fundusz"].apply(is_not_empty) &
        (df["typ_aktywa"].apply(is_not_empty) |
         df["isin"].apply(is_not_empty) |
         df["emitent"].apply(is_not_empty))
    ]

    return ensure_output_schema(df)


def parse_pzu_excel(file_path: str) -> pd.DataFrame:
    """
    Parse TFI PZU Excel files.
    Reads the file, extracts metadata (date), detects header row, and maps columns.
    """
    sheet_name = select_pzu_excel_sheet(
        file_path,
        expected_keywords=["nazwa subfunduszu", "typ instrumentu", "emitent", "kod isin instrumentu"],
    )
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=None)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    # Extract date from first few rows (e.g., "na dzień 31/10/2025")
    extracted_date = ""
    for row_idx in range(min(10, len(df_raw))):
        row = df_raw.iloc[row_idx].tolist()
        for value in row:
            if isinstance(value, str):
                value = value.strip()
                # Look for "na dzień" followed by a date
                if "na dzień" in value.lower():
                    # Try to extract date from the same cell or nearby
                    match = re.search(r"(\d{1,2})[./\s](\d{1,2})[./\s](\d{4})", value)
                    if match:
                        try:
                            day, month, year = match.groups()
                            extracted_date = datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y").strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
            elif isinstance(value, datetime):
                extracted_date = value.strftime("%Y-%m-%d")
                break
        if extracted_date:
            break

    # Detect header row using existing helper
    header_row = detect_allianz_header_row(df_raw)

    # Re-read with detected header
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=header_row)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    # Map PZU-specific columns to OUTPUT_COLUMNS
    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "emitent": "emitent",
        "kod isin instrumentu": "isin",
        "waluta wyceny instrumentu": "waluta",
        "ilość instrumentów w portfelu": "liczba_sztuk",
        "wartość instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "PZU TFI S.A."
    df["fundusz"] = df["fundusz"].apply(normalize_pzu_fundusz)
    df["emitent"] = df["emitent"].apply(_fix_mojibake_text)
    df["typ_aktywa"] = df["typ_aktywa"].apply(_fix_mojibake_text).replace(_PZU_TYPAKTYWA_FIXES)
    df = fix_pzu_shifted_isin_waluta(df)

    # Assign extracted date to the data column
    if extracted_date:
        df["data"] = extracted_date
    else:
        file_date = extract_date_from_filename(file_path)
        df["data"] = file_date

    # Filter out rows where fundusz or isin is empty
    def is_not_empty(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s != "" and s != "nan" and s != "none"

    fundusz_nonempty = df["fundusz"].apply(is_not_empty).fillna(False).astype(bool)
    isin_nonempty = df["isin"].apply(is_not_empty).fillna(False).astype(bool)
    df = df[fundusz_nonempty | isin_nonempty]

    # Filter to keep only PPK/inPZU funds (both naming variants)
    fundusz_col = df.get("fundusz", "").astype(str)
    ppk_mask = fundusz_col.apply(is_pzu_ppk_fund)
    if ppk_mask.any():
        df = df[ppk_mask]
        ppk_inpzu_mask = df["fundusz"].astype(str).str.contains(r"^\s*PPK\s*inPZU\b", case=False, regex=True, na=False)
        puls_mask = df["fundusz"].astype(str).str.contains(r"^\s*inPZU\s*Puls\s*(?:Życia|Zycia)\b", case=False, regex=True, na=False)
        if ppk_inpzu_mask.any() and puls_mask.any():
            df = df[ppk_inpzu_mask]
    else:
        df = df[fundusz_col.str.strip().ne("") & fundusz_col.str.strip().str.lower().ne("nan")]
    
    df = df.dropna(axis=0, how="all")

    return ensure_output_schema(df)


def parse_pzu1_excel(file_path: str) -> pd.DataFrame:
    """
    Parse PZU1 Excel files for 4Q23/4Q24.
    Date is sourced only from filename: PZU1_YYYY-MM-DD.xlsx
    """
    sheet_name = select_pzu_excel_sheet(
        file_path,
        expected_keywords=["fundusz", "typ aktywa", "emitent", "isin", "wartość"],
    )
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=0)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    def _find_col(candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            col = find_column(df, normalize_header(candidate))
            if col:
                return col
        return None

    fund_col = _find_col([
        "fundusz",
        "nazwa subfunduszu",
        "nazwa funduszu",
    ])
    typ_col = _find_col([
        "typ_aktywa",
        "typ instrumentu",
        "rodzaj instrumentu",
    ])
    emitent_col = _find_col([
        "emitent",
        "nazwa emitenta",
    ])
    isin_col = _find_col([
        "isin",
        "kod isin",
        "kod isin instrumentu",
        "identyfikator instrumentu",
    ])
    waluta_col = _find_col([
        "waluta wyceny instrumentu",
        "waluta instrumentu",
        "waluta",
    ])
    liczba_col = _find_col([
        "liczba_sztuk",
        "ilość instrumentów w portfelu",
        "ilość",
        "ilosc",
    ])
    wartosc_col = _find_col([
        "wartosc_pln",
        "wartosc_wg_wyceny",
        "wartość wg wyceny",
        "wartość instrumentu w walucie wyceny funduszu",
        "wartość instrumentu",
        "wartość",
    ])
    nominal_col = _find_col([
        "wartosc_nominalna",
        "wartość nominalna",
    ])

    df["fundusz"] = df[fund_col] if fund_col else ""
    df["typ_aktywa"] = df[typ_col] if typ_col else ""
    df["emitent"] = df[emitent_col] if emitent_col else ""
    df["waluta"] = df[waluta_col] if waluta_col else ""
    df["liczba_sztuk"] = df[liczba_col] if liczba_col else ""
    df["wartosc_pln"] = df[wartosc_col] if wartosc_col else ""

    if wartosc_col:
        wartosc_num = df["wartosc_pln"].apply(parse_polish_number)
        finite = wartosc_num.dropna()
        if not finite.empty:
            max_abs = finite.abs().max()
            sum_abs = finite.abs().sum()
            wartosc_col_norm = normalize_header(wartosc_col)
            file_name = os.path.basename(file_path).lower()
            # Scale only when source header explicitly declares thousand-units.
            # PZU1 files (4Q23/4Q24 in this project) are reported in thousands,
            # so force thousand-scaling for those files.
            declared_thousands = bool(re.search(r"\b(tys|tys\.|tysiac|tysiace)\b", wartosc_col_norm))
            pzu1_force_thousands = file_name.startswith("pzu1_")
            if (declared_thousands or pzu1_force_thousands) and max_abs < 1_000_000 and sum_abs < 20_000_000:
                df["wartosc_pln"] = (wartosc_num * 1000).apply(format_decimal_comma)

    if isin_col:
        isin_series = df[isin_col].astype(str).str.strip()
        df["_isin_raw"] = isin_series
        df["isin"] = isin_series.str.extract(r"([A-Z]{2}[A-Z0-9]{10})", expand=False).fillna("")
    else:
        df["_isin_raw"] = ""
        df["isin"] = ""

    if nominal_col:
        waluta_series = df["waluta"].astype(str).str.strip()
        waluta_is_isin = waluta_series.str.match(r"^[A-Z]{2}[A-Z0-9]{10}$", na=False)

        qty_series = df["liczba_sztuk"].astype(str).str.strip()
        qty_numeric = df["liczba_sztuk"].apply(parse_polish_number)
        qty_missing_or_invalid = qty_numeric.isna()

        waluta_from_qty = qty_series.str.upper()
        waluta_like = waluta_from_qty.str.match(r"^[A-Z]{3,6}$", na=False)
        waluta_fill_mask = waluta_is_isin & waluta_like
        if waluta_fill_mask.any():
            df.loc[waluta_fill_mask, "waluta"] = waluta_from_qty[waluta_fill_mask]

        nominal_series = df[nominal_col]
        nominal_text = nominal_series.astype(str).str.strip()
        nominal_present = ~(nominal_text.str.lower().isin(["", "nan", "none"]) | nominal_series.isna())

        qty_backfill_mask = qty_missing_or_invalid & waluta_is_isin & nominal_present
        if qty_backfill_mask.any():
            df.loc[qty_backfill_mask, "liczba_sztuk"] = nominal_series[qty_backfill_mask]

    file_date = extract_date_from_filename(file_path)
    df["instytucja"] = "PZU TFI S.A."
    df["data"] = file_date if file_date else ""
    df["fundusz"] = df["fundusz"].apply(normalize_pzu_fundusz)
    df["emitent"] = df["emitent"].apply(_fix_mojibake_text)
    df["typ_aktywa"] = df["typ_aktywa"].apply(_fix_mojibake_text).replace(_PZU_TYPAKTYWA_FIXES)

    emitent_series = df["emitent"].astype(str).str.strip()
    emitent_blank = emitent_series.str.lower().isin(["", "nan", "none", "0"])
    waluta_series = df["waluta"].astype(str).str.strip()
    waluta_is_isin = waluta_series.str.match(r"^[A-Z]{2}[A-Z0-9]{10}$", na=False)
    isin_raw_series = df["_isin_raw"].astype(str).str.strip()
    isin_raw_valid_text = ~isin_raw_series.str.lower().isin(["", "nan", "none", "0"])
    fill_emitent_from_raw_shift = emitent_blank & waluta_is_isin & isin_raw_valid_text
    if fill_emitent_from_raw_shift.any():
        df.loc[fill_emitent_from_raw_shift, "emitent"] = isin_raw_series[fill_emitent_from_raw_shift]

    emitent_series = df["emitent"].astype(str).str.strip()
    emitent_blank = emitent_series.str.lower().isin(["", "nan", "none", "0"])
    isin_raw_has_isin = isin_raw_series.str.contains(r"\b[A-Z]{2}[A-Z0-9]{10}\b", na=False)
    emitent_from_isin_raw = (
        isin_raw_series
        .str.replace(r"\b[A-Z]{2}[A-Z0-9]{10}\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(" ,;-")
    )
    fill_emitent_from_isin = emitent_blank & isin_raw_has_isin & emitent_from_isin_raw.ne("")
    if fill_emitent_from_isin.any():
        df.loc[fill_emitent_from_isin, "emitent"] = emitent_from_isin_raw[fill_emitent_from_isin]

    df = fix_pzu_shifted_isin_waluta(df)
    df = df.drop(columns=["_isin_raw"], errors="ignore")

    fundusz_series = df["fundusz"].astype(str)
    ppk_mask = fundusz_series.apply(is_pzu_ppk_fund)
    if ppk_mask.any():
        df = df[ppk_mask]
    else:
        df = df[fundusz_series.str.strip().ne("") & fundusz_series.str.strip().str.lower().ne("nan")]

    df = apply_pzu_eop_2023_correction(df, file_path)

    file_name = os.path.basename(file_path).lower()
    if file_name == "pzu1_2023-12-31.xlsx" and not df.empty:
        target_years = set(_PZU_EOP_2023_VALUES.keys())
        year_norm = (
            df["DATA_fundusz"]
            .astype(str)
            .str.extract(r"(20\d{2})", expand=False)
            .fillna("")
        )
        pzu_mask = (
            year_norm.isin(target_years)
            & (
                df.get("isin", "").astype(str).str.upper().str.strip().eq("PLPZU0000011")
                | df.get("emitent", "").astype(str).str.contains(r"\bPZU\b", case=False, regex=True, na=False)
            )
        )

        template = {col: "" for col in df.columns}
        if not df.empty:
            sample = df.iloc[0].to_dict()
            template.update({k: ("" if pd.isna(v) else v) for k, v in sample.items()})

        df = df.loc[~pzu_mask].copy()
        hard_rows: List[Dict[str, Any]] = []
        for year, corrected_value in _PZU_EOP_2023_VALUES.items():
            row = dict(template)
            row["fundusz"] = f"inPZU Puls Życia {year}"
            row["DATA_fundusz"] = year
            row["typ_aktywa"] = "Akcje"
            row["emitent"] = "PZU"
            row["isin"] = "PLPZU0000011"
            row["waluta"] = "PLN"
            row["liczba_sztuk"] = ""
            row["wartosc_pln"] = corrected_value
            row["TYP_aktywo_std"] = "akcje"
            hard_rows.append(row)
        df = pd.concat([df, pd.DataFrame(hard_rows)], ignore_index=True)

    # Prevent textual NaN values from leaking to CSV output.
    for col in ("wartosc_pln", "liczba_sztuk", "isin", "waluta", "emitent"):
        if col in df.columns:
            series = df[col]
            df[col] = series.where(~series.isna(), "")
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "NaN": "", "None": "", "NONE": ""})
            )

    df = df.dropna(axis=0, how="all")

    return ensure_output_schema(df)


def parse_pzu2_excel(file_path: str) -> pd.DataFrame:
    """
    Parse PZU2 Excel files for 4Q22.
    Required mapping:
    - data: 2022-12-31
    - instytucja: PZU TFI S.A.
    - wartosc_pln: source value in thousands of PLN, multiplied by 1000
    - isin: extract only value inside parentheses; if no parentheses -> empty
    """
    sheet_name = select_pzu_excel_sheet(
        file_path,
        expected_keywords=["fundusz", "typ aktywa", "emitent", "isin", "wartość"],
    )
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", header=0)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    def _find_col(candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            col = find_column(df, normalize_header(candidate))
            if col:
                return col
        return None

    fund_col = _find_col(["fundusz", "nazwa subfunduszu", "nazwa funduszu"])
    typ_col = _find_col(["typ_aktywa", "typ instrumentu", "rodzaj instrumentu"])
    emitent_col = _find_col(["emitent", "nazwa emitenta"])
    isin_col = _find_col(["isin", "kod isin", "kod isin instrumentu", "identyfikator instrumentu"])
    waluta_col = _find_col(["waluta", "waluta instrumentu", "waluta wyceny instrumentu"])
    liczba_col = _find_col(["liczba_sztuk", "liczba sztuk", "ilość", "ilosc"])
    wartosc_col = _find_col([
        "wartosc_pln (tys. zł)",
        "wartosc_pln (tys. zl)",
        "wartosc_pln",
        "wartość instrumentu w walucie wyceny funduszu",
    ])

    out = pd.DataFrame(index=df.index)
    out["fundusz"] = df[fund_col] if fund_col else ""
    out["typ_aktywa"] = df[typ_col] if typ_col else ""
    out["emitent"] = df[emitent_col] if emitent_col else ""
    out["waluta"] = df[waluta_col] if waluta_col else ""
    out["liczba_sztuk"] = df[liczba_col] if liczba_col else ""

    raw_isin = df[isin_col].astype(str).str.strip() if isin_col else pd.Series([""] * len(df), index=df.index)
    out["isin"] = raw_isin

    wartosc_num = (df[wartosc_col].apply(parse_polish_number) if wartosc_col else pd.Series([None] * len(df), index=df.index))
    out["wartosc_pln"] = wartosc_num * 1000

    out["data"] = "2022-12-31"
    out["instytucja"] = "PZU TFI S.A."

    out = ensure_output_schema(out)
    out["isin"] = out["isin"].replace("nan", "")
    return out


def parse_esaliens_text_file(file_path: str) -> pd.DataFrame:
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(file_path))
    file_date = date_match.group(1) if date_match else ""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return ensure_output_schema(pd.DataFrame())
    return _parse_esaliens_texts([text], file_date)


def _parse_esaliens_texts(texts: List[str], file_date: str) -> pd.DataFrame:
    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    number_pattern = re.compile(r"-?\d[\d\s]*,\d{2}")
    percent_pattern = re.compile(r"(?<![\d,])-?\d[\d\s]*,\d{2}%$")

    rows: List[Dict[str, str]] = []

    def _normalize_line(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"-\s+(?=\d)", "-", text)

    def _extract_row(line: str) -> Optional[Dict[str, str]]:
        line = _normalize_line(line)
        percent_match = percent_pattern.search(line)
        if not percent_match:
            return None

        line_wo_pct = line[:percent_match.start()].strip()
        numbers = number_pattern.findall(line_wo_pct)
        if len(numbers) < 2:
            return None
        liczba_sztuk = numbers[-2]
        wartosc_pln = numbers[-1]
        line_tail_stripped = re.sub(
            rf"{re.escape(liczba_sztuk)}\s+{re.escape(wartosc_pln)}\s*$",
            "",
            line_wo_pct,
        ).strip()
        tokens = line_tail_stripped.split()
        if not tokens:
            return None

        curr_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if re.fullmatch(r"[A-Z]{3}", tokens[i]):
                curr_idx = i
                break
        if curr_idx is None or curr_idx < 1:
            return None
        waluta = tokens[curr_idx]
        kraj = tokens[curr_idx - 1] if re.fullmatch(r"[A-Z]{2}", tokens[curr_idx - 1]) else ""
        if not kraj:
            return None

        isin_idx = None
        for i in range(curr_idx - 2, -1, -1):
            if isin_pattern.fullmatch(tokens[i]) or tokens[i] == "N/D":
                isin_idx = i
                break
        if isin_idx is None:
            return None
        isin = tokens[isin_idx]
        typ_aktywa = " ".join(tokens[isin_idx + 1: curr_idx - 1]).strip()

        fundusz = ""
        fundusz_match = re.search(
            r"(Esaliens PPK Specjalistyczny Fundusz Inwestycyjny Otwarty\s+ESA\s+\d{4}\s+SFIO)",
            line,
            flags=re.IGNORECASE,
        )
        if fundusz_match:
            fundusz = fundusz_match.group(1).strip()
        else:
            fundusz_matches = re.findall(r"(Esaliens[^\n]*?SFIO)", line, flags=re.IGNORECASE)
            if fundusz_matches:
                fundusz = fundusz_matches[-1].strip()

        emitent = ""
        if fundusz:
            lower_line = line.lower()
            pos_fundusz = lower_line.find(fundusz.lower())
            if pos_fundusz >= 0:
                after_fundusz = line[pos_fundusz + len(fundusz):]
                pos_isin = after_fundusz.find(isin)
                if pos_isin >= 0:
                    emitent = after_fundusz[:pos_isin].strip()
        if not emitent:
            emitent = " ".join(tokens[:isin_idx]).strip()
            if fundusz:
                emitent = emitent.replace(fundusz, "").strip()

        if not emitent or not typ_aktywa or not waluta:
            return None

        return {
            "data": file_date,
            "instytucja": "ESALIENS TFI S.A.",
            "fundusz": fundusz,
            "typ_aktywa": typ_aktywa,
            "emitent": emitent,
            "isin": isin,
            "waluta": waluta,
            "liczba_sztuk": liczba_sztuk,
            "wartosc_pln": wartosc_pln,
        }

    for text in texts:
        if not text:
            continue
        buffer = ""
        for raw_line in text.splitlines():
            line = _normalize_line(raw_line)
            if not line:
                continue
            buffer = f"{buffer} {line}".strip() if buffer else line
            if percent_pattern.search(buffer) and len(number_pattern.findall(buffer)) >= 2:
                row = _extract_row(buffer)
                if row:
                    rows.append(row)
                buffer = ""

        if buffer:
            row = _extract_row(buffer)
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_text = pd.DataFrame(rows)
    df_text = ensure_output_schema(df_text)
    return df_text.drop_duplicates(subset=OUTPUT_COLUMNS)


def parse_esaliens_pdf(file_path: str) -> pd.DataFrame:
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(file_path))
    file_date = date_match.group(1) if date_match else ""
    pdf_path = file_path
    cleaned_path = None

    try:
        raw_bytes = None
        with open(file_path, "rb") as fh:
            raw_bytes = fh.read()
        if raw_bytes:
            pdf_start = raw_bytes.find(b"%PDF-")
            pdf_end = raw_bytes.rfind(b"%%EOF")
            if pdf_start > 0 or (pdf_end != -1 and pdf_end + 5 < len(raw_bytes)):
                cleaned_bytes = raw_bytes[pdf_start:pdf_end + 5] if pdf_start != -1 and pdf_end != -1 else raw_bytes
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(cleaned_bytes)
                    cleaned_path = tmp.name
                    pdf_path = cleaned_path
    except Exception:
        pdf_path = file_path

    def safe_page_text_simple(page) -> str:
        extract_simple = getattr(page, "extract_text_simple", None)
        extract_fallback = getattr(page, "extract_text", None)

        def _timeout_handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.alarm(8)
            text = extract_simple() if callable(extract_simple) else ""
            if text:
                return text
            if callable(extract_fallback):
                return extract_fallback() or ""
            return ""
        except TimeoutError:
            return ""
        except Exception:
            return ""
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    page_texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = safe_page_text_simple(page)
            page_texts.append(page_text or "")

    if not any(text.strip() for text in page_texts) and pytesseract and convert_from_path:
        ocr_texts: List[str] = []

        def _ocr_pdf(path: str) -> List[str]:
            texts: List[str] = []
            for page_number in range(1, len(page_texts) + 1):
                images = convert_from_path(
                    path,
                    dpi=200,
                    first_page=page_number,
                    last_page=page_number,
                )
                for img in images:
                    texts.append(pytesseract.image_to_string(img, lang="pol") or "")
            return texts

        try:
            ocr_texts = _ocr_pdf(pdf_path)
        except Exception:
            ocr_texts = []

        if not any(text.strip() for text in ocr_texts) and shutil.which("gs"):
            repaired_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    repaired_path = tmp.name
                subprocess.run(
                    [
                        "gs",
                        "-q",
                        "-dNOPAUSE",
                        "-dBATCH",
                        "-sDEVICE=pdfwrite",
                        f"-sOutputFile={repaired_path}",
                        pdf_path,
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                ocr_texts = _ocr_pdf(repaired_path)
            finally:
                if repaired_path:
                    try:
                        os.remove(repaired_path)
                    except Exception:
                        pass

        page_texts = ocr_texts
    if cleaned_path:
        try:
            os.remove(cleaned_path)
        except Exception:
            pass

    df_text = _parse_esaliens_texts(page_texts, file_date)
    if not df_text.empty:
        return df_text

    tables = extract_tables_from_pdf(file_path, max_pages=None)
    if not tables:
        return ensure_output_schema(pd.DataFrame())

    df = pd.concat(tables, ignore_index=True, sort=False)

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta": "emitent",
        "kod isin instrumentu": "isin",
        "waluta wykorzystywana do wyceny instrumentu": "waluta",
        "ilość instrumentu w portfelu": "liczba_sztuk",
        "wartość instrumentu w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = "ESALIENS TFI S.A."
    df["data"] = file_date
    df = df.dropna(axis=0, how="all")
    def _is_blank(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().isin(["", "nan", "none"])

    df["_liczba_sztuk_num"] = df.get("liczba_sztuk", "").apply(parse_polish_number)
    df["_wartosc_pln_num"] = df.get("wartosc_pln", "").apply(parse_polish_number)
    df = df[
        (~_is_blank(df.get("fundusz", "")))
        | (~_is_blank(df.get("typ_aktywa", "")))
        | (~_is_blank(df.get("emitent", "")))
        | (~_is_blank(df.get("isin", "")))
        | (~_is_blank(df.get("waluta", "")))
        | (df["_liczba_sztuk_num"] != 0)
        | (df["_wartosc_pln_num"] != 0)
    ].drop(columns=["_liczba_sztuk_num", "_wartosc_pln_num"], errors="ignore")
    df = ensure_output_schema(df)

    def _is_blank_final(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().isin(["", "nan", "none"])

    liczba_sztuk_num = df["liczba_sztuk"].apply(parse_polish_number)
    wartosc_pln_num = df["wartosc_pln"].apply(parse_polish_number)

    df = df[
        (~_is_blank_final(df["fundusz"]))
        | (~_is_blank_final(df["typ_aktywa"]))
        | (~_is_blank_final(df["emitent"]))
        | (~_is_blank_final(df["isin"]))
        | (~_is_blank_final(df["waluta"]))
        | (liczba_sztuk_num != 0)
        | (wartosc_pln_num != 0)
    ]
    return df


def parse_esaliens_text(file_path: str) -> pd.DataFrame:
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(file_path))
    file_date = date_match.group(1) if date_match else ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    if not text.strip():
        return ensure_output_schema(pd.DataFrame())

    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    number_pattern = re.compile(r"-?\d[\d\s]*,\d{2}")
    percent_pattern = re.compile(r"(?<![\d,])-?\d[\d\s]*,\d{2}%$")

    rows: List[Dict[str, str]] = []

    def _normalize_line(line: str) -> str:
        line = re.sub(r"\s+", " ", line).strip()
        return re.sub(r"-\s+(?=\d)", "-", line)

    def _extract_row(line: str) -> Optional[Dict[str, str]]:
        line = _normalize_line(line)
        percent_match = percent_pattern.search(line)
        if not percent_match:
            return None

        line_wo_pct = line[:percent_match.start()].strip()
        numbers = number_pattern.findall(line_wo_pct)
        if len(numbers) < 2:
            return None

        liczba_sztuk = numbers[-2]
        wartosc_pln = numbers[-1]
        line_tail_stripped = re.sub(
            rf"{re.escape(liczba_sztuk)}\s+{re.escape(wartosc_pln)}\s*$",
            "",
            line_wo_pct,
        ).strip()
        tokens = line_tail_stripped.split()
        if not tokens:
            return None

        curr_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if re.fullmatch(r"[A-Z]{3}", tokens[i]):
                curr_idx = i
                break
        if curr_idx is None or curr_idx < 1:
            return None
        waluta = tokens[curr_idx]
        kraj = tokens[curr_idx - 1] if re.fullmatch(r"[A-Z]{2}", tokens[curr_idx - 1]) else ""
        if not kraj:
            return None

        isin_idx = None
        for i in range(curr_idx - 2, -1, -1):
            if isin_pattern.fullmatch(tokens[i]) or tokens[i] == "N/D":
                isin_idx = i
                break
        if isin_idx is None:
            return None

        isin = tokens[isin_idx]
        typ_aktywa = " ".join(tokens[isin_idx + 1: curr_idx - 1]).strip()

        fundusz = ""
        fundusz_matches = re.findall(r"(Esaliens[^\n]*?ESA\s+\d{4}\s+SFIO)", line, flags=re.IGNORECASE)
        if not fundusz_matches:
            fundusz_matches = re.findall(r"(Esaliens[^\n]*?SFIO)", line, flags=re.IGNORECASE)
        if fundusz_matches:
            fundusz = fundusz_matches[-1].strip()
        if not fundusz:
            return None

        emitent = ""
        lower_line = line.lower()
        pos_fundusz = lower_line.find(fundusz.lower())
        if pos_fundusz >= 0:
            after_fundusz = line[pos_fundusz + len(fundusz):]
            pos_isin = after_fundusz.find(isin)
            if pos_isin >= 0:
                emitent = after_fundusz[:pos_isin].strip()
        if not emitent:
            emitent = " ".join(tokens[:isin_idx]).strip()
            emitent = emitent.replace(fundusz, "").strip()

        if not emitent or not typ_aktywa or not waluta:
            return None

        return {
            "data": file_date,
            "instytucja": "ESALIENS TFI S.A.",
            "fundusz": fundusz,
            "typ_aktywa": typ_aktywa,
            "emitent": emitent,
            "isin": isin,
            "waluta": waluta,
            "liczba_sztuk": liczba_sztuk,
            "wartosc_pln": wartosc_pln,
        }

    buffer = ""
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        if percent_pattern.fullmatch(line) and buffer:
            buffer = f"{buffer} {line}".strip()
        else:
            buffer = f"{buffer} {line}".strip() if buffer else line

        if percent_pattern.search(buffer) and len(number_pattern.findall(buffer)) >= 2:
            row = _extract_row(buffer)
            if row:
                rows.append(row)
            buffer = ""

    if buffer:
        row = _extract_row(buffer)
        if row:
            rows.append(row)

    if not rows:
        return ensure_output_schema(pd.DataFrame())

    df_text = pd.DataFrame(rows)
    df_text = ensure_output_schema(df_text)
    df_text = df_text.drop_duplicates(subset=OUTPUT_COLUMNS)
    return df_text


def parse_pekao_excel(file_path: str) -> pd.DataFrame:
    """Parse Pekao TFI S.A. Excel files for PPK holdings."""
    file_name = os.path.basename(file_path)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", file_name)
    data = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else ""
    instytucja = "Pekao TFI S.A."

    try:
        df = pd.read_excel(file_path, engine="openpyxl", header=1)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    df.columns = [normalize_header(c) for c in df.columns]

    mapping = {
        "nazwa funduszu lub subfunduszu": "fundusz",
        "typ instrumentu": "typ_aktywa",
        "nazwa emitenta instrumentu": "emitent",
        "kod isin składnika portfela funduszu lub subfunduszu": "isin",
        "waluta instrumentu składnika lokat funduszu lub subfunduszu": "waluta",
        "ilość składnika lokat funduszu lub subfunduszu": "liczba_sztuk",
        "wartość składnika lokat w walucie wyceny funduszu lub subfunduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        df[dst] = df[col] if col else ""

    df["instytucja"] = instytucja
    df["data"] = data

    df["fundusz"] = df["fundusz"].astype(str).str.strip()
    df = df[df["fundusz"].str.contains(r"PPK", case=False, na=False)].copy()
    if df.empty:
        return ensure_output_schema(pd.DataFrame())

    # Normalize columns to source-like strings, preserving Excel numeric formatting
    df["liczba_sztuk"] = df["liczba_sztuk"].apply(lambda v: str(v).replace(" ", "").replace(",", ".") if pd.notna(v) else "")
    df["wartosc_pln"] = df["wartosc_pln"].apply(lambda v: str(v).replace(" ", "").replace(",", ".") if pd.notna(v) else "")

    return ensure_output_schema(df)


def parse_pekao_pdf(file_path: str) -> pd.DataFrame:
    """
    Parse Pekao TFI S.A. PDF files.
    Extracts portfolio holdings from raw text format (full line format).
    Format: .X.Y PIO### ISIN FundName 0 FIO PLN CIC Emitent ISIN Code Country Type Category Country Country Currency Qty Value %
    """
    instytucja = "Pekao TFI S.A."
    
    # Extract date from filename (format: Pekao_YYYY-MM-DD.pdf)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(file_path))
    if date_match:
        data = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        data = ""
    
    rows: List[Dict[str, str]] = []
    current_fund = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                
                for line in page_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Match lines starting with .X.Y or X.Y.Z format
                    if not re.match(r'^\.?\d+\.\d+', line):
                        continue
                    
                    # Skip lines without enough data
                    if len(line) < 50:
                        continue
                    
                    # Find all 12-character ISINs (format: 2 letters + 10 alphanumeric)
                    isins = re.findall(r'\b([A-Z]{2}[A-Z0-9]{10})\b', line)
                    if len(isins) < 2:
                        # Need at least fund ISIN and instrument ISIN
                        continue
                    
                    # Split line into parts
                    parts = line.split()
                    
                    # Extract fund ISIN (should be parts[2])
                    fund_isin = parts[2] if len(parts) > 2 and re.match(r'^[A-Z]{2}[A-Z0-9]{10}$', parts[2]) else ""
                    
                    # Extract fund name - search for "Pekao" keyword
                    fund_name_parts = []
                    fund_start = -1
                    for i, part in enumerate(parts):
                        if 'Pekao' in part:
                            fund_start = i
                            break
                    
                    if fund_start >= 0:
                        # Collect fund name until we hit "0" or "FIO" or "SFIO"
                        for i in range(fund_start, len(parts)):
                            if parts[i] in ['0', 'FIO', 'SFIO'] or re.match(r'^[A-Z]{3}\d+$', parts[i]):
                                break
                            fund_name_parts.append(parts[i])
                        
                        if fund_name_parts:
                            current_fund = ' '.join(fund_name_parts)
                    
                    if not current_fund:
                        continue
                    
                    # Skip non-PPK funds (only process funds with "PPK" in name)
                    if 'PPK' not in current_fund:
                        continue
                    
                    # Find instrument ISIN (should be the second ISIN found, not the fund ISIN)
                    instrument_isin = ""
                    for isin in isins:
                        if isin != fund_isin:
                            instrument_isin = isin
                            break
                    
                    if not instrument_isin:
                        continue
                    
                    # Find ISIN position in parts
                    isin_idx = -1
                    for i, part in enumerate(parts):
                        if part == instrument_isin:
                            isin_idx = i
                            break
                    
                    if isin_idx < 0:
                        continue
                    
                    # Emitent is between CIC code (XL##, etc.) and instrument ISIN
                    # Find CIC pattern (XL##, PL##, DE##, etc.)
                    cic_idx = -1
                    for i in range(len(parts)):
                        if re.match(r'^[A-Z]{2}\d+$', parts[i]) and i < isin_idx:
                            cic_idx = i
                            break
                    
                    emitent_parts = []
                    if cic_idx >= 0 and cic_idx < isin_idx - 1:
                        emitent_parts = parts[cic_idx + 1:isin_idx]
                    
                    emitent = ' '.join(emitent_parts) if emitent_parts else ""
                    
                    # Type of instrument: after ISIN + code + country code
                    # Structure: ISIN Code Country# Type Category# Country Country Currency
                    # Example: PL0000113783 DS0432 PL11 obligacje skarbowe 1 PL PL PLN
                    # Example: PLALIOR00045 ALIOR PL31 akcje zwykłe 3L PL PL PLN
                    # Example: PLBRE0005227 MBK 10.63 PERPPL25 obligacje korporacyjne 2 PL PL EUR
                    typ_parts = []
                    if isin_idx + 3 < len(parts):
                        # Start after ISIN, Code, Country#
                        start_idx = isin_idx + 3
                        # Collect typ until we hit a short code (2-3 chars like "3L", "1", etc.) or currency
                        for i in range(start_idx, len(parts)):
                            part = parts[i]
                            # Skip numbers (like "10.63")
                            if re.match(r'^[\d,.]+$', part):
                                continue
                            # REIT is a valid asset type, not a code
                            if part.upper() == "REIT":
                                typ_parts.append("REIT")
                                continue
                            # Skip short codes that are all uppercase/digits (like "PERPPL25", "3L")
                            if re.match(r'^[A-Z0-9]{4,}$', part):
                                continue
                            # Stop at: single digit, very short code, or currency
                            if re.match(r'^\d+$', part) or re.match(r'^[A-Z0-9]{1,2}$', part.upper()) or part in ['PLN', 'EUR', 'USD', 'GBP', 'CHF', 'CZK', 'HUF', 'RON', 'CAD']:
                                break
                            typ_parts.append(part)
                    
                    # Clean and capitalize typ_aktywa
                    typ_aktywa = ' '.join(typ_parts).strip() if typ_parts else ""
                    # Capitalize first letter of each word for consistency
                    if typ_aktywa:
                        typ_aktywa = ' '.join(word.capitalize() for word in typ_aktywa.split())
                    
                    # Find currency (PLN, EUR, USD, etc.) - should appear multiple times
                    # The last occurrence before numbers is the instrument currency
                    currencies = ['PLN', 'EUR', 'USD', 'GBP', 'CHF', 'CZK', 'HUF', 'RON']
                    waluta = ""
                    waluta_positions = []
                    for i, part in enumerate(parts):
                        if part in currencies:
                            waluta_positions.append(i)
                    
                    if waluta_positions:
                        # Use the last currency occurrence (instrument currency, not fund currency)
                        waluta_idx = waluta_positions[-1]
                        waluta = parts[waluta_idx]
                    else:
                        continue
                    
                    # Numbers after currency: quantity, value, percentage
                    # Format: "72" "750." "60" "119" "145.00" "8.45%"
                    # Quantity ends with ".", value ends with ".XX", percentage has %
                    numbers = []
                    for i in range(waluta_idx + 1, len(parts)):
                        part = parts[i]
                        # Check if it's a number-like token (with or without %)
                        if re.match(r'^[\d,.]+%?$', part):
                            numbers.append(part)
                    
                    # Parse numbers: find quantity (ends with single "."), value (ends with ".XX"), % (last)
                    liczba_sztuk = ""
                    wartosc_pln = ""
                    
                    if len(numbers) >= 2:
                        # Remove percentage (last item with %)
                        if '%' in numbers[-1]:
                            numbers = numbers[:-1]
                        
                        # Find where quantity ends (first token ending with single ".")
                        qty_end = -1
                        for i, num in enumerate(numbers):
                            if num.endswith('.') and not re.match(r'.*\.\d+', num):
                                qty_end = i
                                break
                        
                        if qty_end >= 0:
                            # Quantity is from start to qty_end (inclusive)
                            qty_parts = numbers[:qty_end + 1]
                            # Value is the rest
                            value_parts = numbers[qty_end + 1:]
                            
                            liczba_sztuk = ''.join(qty_parts).replace('.', '').replace(',', '.')
                            wartosc_pln = ''.join(value_parts).replace(',', '.')
                        else:
                            # No quantity with single ".", assume all is value
                            wartosc_pln = ''.join(numbers).replace(',', '.')
                    
                    # Clean up
                    if emitent and instrument_isin and wartosc_pln:
                        # Clean fund name: remove trailing " #" pattern
                        # "Pekao PPK 2020 Spokojne Jutro 5" -> "Pekao PPK 2020 Spokojne Jutro"
                        # "Pekao Zrównoważony 1" -> "Pekao Zrównoważony"
                        clean_fund = current_fund
                        if clean_fund:
                            # Remove trailing single digit with space
                            clean_fund = re.sub(r'\s+\d$', '', clean_fund).strip()
                        
                        rows.append({
                            "data": data,
                            "instytucja": instytucja,
                            "fundusz": clean_fund,
                            "typ_aktywa": typ_aktywa,
                            "emitent": emitent.strip(),
                            "isin": instrument_isin,
                            "waluta": waluta,
                            "liczba_sztuk": liczba_sztuk.replace('.', '') if liczba_sztuk else "",
                            "wartosc_pln": wartosc_pln,
                        })
    except Exception:
        pass
    
    df = pd.DataFrame(rows)
    return ensure_output_schema(df)


def parse_investors_pdf(file_path: str) -> pd.DataFrame:
    instytucja = "Investors TFI S.A."

    def clean_number(value: str) -> str:
        if not value:
            return ""
        return value.replace(" ", "").replace(".", "")

    rows: List[Dict[str, str]] = []
    current_fundusz: Optional[str] = None
    currency_pattern = re.compile(r"^[A-Z]{3}$")

    def _append_no_isin_forward(line: str) -> bool:
        match = re.match(
            r"^\d+\s+(.+?)\s+-\s+(.+?)\s+([A-Z]{3})\s+(-?(?:\d{1,3}(?: \d{3})*|\d+))\s+(-?(?:\d{1,3}(?: \d{3})*,\d+|\d+,\d+))\s+[\d,]+\s*%$",
            line,
        )
        if not match or not current_fundusz:
            return False

        emitent, typ_aktywa, waluta, liczba_sztuk, wartosc_pln = match.groups()
        rows.append(
            {
                "data": data,
                "instytucja": instytucja,
                "fundusz": current_fundusz,
                "typ_aktywa": typ_aktywa.strip(),
                "emitent": emitent.strip(),
                "isin": "",
                "waluta": waluta.strip(),
                "liczba_sztuk": clean_number(liczba_sztuk),
                "wartosc_pln": clean_number(wartosc_pln),
            }
        )
        return True

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        sample_text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
        date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", sample_text)
        if date_match:
            data = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
        else:
            data = parse_date_from_text(sample_text) or ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            lines = page_text.split("\n")

            page_fundusz = None
            matches = list(
                re.finditer(
                    r"Subfundusz\s+(.+?)(?:\s+Skład portfela|\s+Strona|$)",
                    page_text,
                )
            )
            if matches:
                potential_fund = matches[-1].group(1).strip()
                if "Strona" not in potential_fund and len(potential_fund) > 5:
                    page_fundusz = potential_fund

            if page_fundusz:
                current_fundusz = page_fundusz

            for line in lines:
                line = line.strip()

                percent_match = re.search(r"-?\d+,\d+%", line)
                if percent_match:
                    line = line[: percent_match.end()]

                if any(
                    skip in line
                    for skip in [
                        "Lp. Nazwa emitenta",
                        "Ze względu na",
                        "Investor PPK SFIO",
                        "Strona",
                    ]
                ):
                    continue

                if line.startswith("Subfundusz"):
                    continue

                if not re.match(r"^\d+\s+", line):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                isin_idx = None
                for idx, part in enumerate(parts):
                    if re.match(r"^[A-Z]{2}[A-Z0-9]{10}$", part) and any(
                        c.isdigit() for c in part
                    ):
                        isin_idx = idx
                        break

                if isin_idx is None:
                    if _append_no_isin_forward(line):
                        continue
                    continue

                if isin_idx < 2:
                    continue

                emitent = " ".join(parts[1:isin_idx])
                isin = parts[isin_idx]

                remaining = parts[isin_idx + 1 :]
                if len(remaining) < 4:
                    continue

                if not remaining[-1].endswith("%"):
                    continue

                comma_idx = None
                for j in range(len(remaining) - 2, -1, -1):
                    if "," in remaining[j]:
                        comma_idx = j
                        break

                if comma_idx is None:
                    continue

                waluta = None
                waluta_idx = -1
                for j in range(comma_idx - 1, -1, -1):
                    if currency_pattern.match(remaining[j]):
                        waluta = remaining[j]
                        waluta_idx = j
                        break

                if waluta is None or waluta_idx < 0:
                    continue

                numeric_after_currency = [
                    token
                    for token in remaining[waluta_idx + 1 : comma_idx]
                    if re.fullmatch(r"-?\d+", token)
                ]

                if len(numeric_after_currency) <= 1:
                    qty_token_count = len(numeric_after_currency)
                else:
                    qty_token_count = min(2, len(numeric_after_currency) - 1)

                qty_tokens = numeric_after_currency[:qty_token_count]
                value_int_tokens = numeric_after_currency[qty_token_count:]

                if value_int_tokens:
                    wartosc_pln = " ".join(value_int_tokens + [remaining[comma_idx]])
                else:
                    wartosc_pln = remaining[comma_idx]

                liczba_sztuk = " ".join(qty_tokens)

                typ_aktywa = remaining[0]

                liczba_sztuk = clean_number(liczba_sztuk)
                wartosc_pln = clean_number(wartosc_pln)

                if current_fundusz and emitent and isin:
                    rows.append(
                        {
                            "data": data,
                            "instytucja": instytucja,
                            "fundusz": current_fundusz,
                            "typ_aktywa": typ_aktywa,
                            "emitent": emitent,
                            "isin": isin,
                            "waluta": waluta,
                            "liczba_sztuk": liczba_sztuk,
                            "wartosc_pln": wartosc_pln,
                        }
                    )

    df = pd.DataFrame(rows)
    return ensure_output_schema(df)


def parse_uniqa_pdf(file_path: str) -> pd.DataFrame:
    file_date = extract_date_from_filename(file_path)
    if not file_date:
        preview_text = extract_text_from_pdf(file_path, max_pages=1)
        file_date = parse_date_from_text(preview_text) or ""

    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
    }
    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    isin_loose_pattern = re.compile(r"(?:[A-Z][^A-Z0-9]*){2}(?:[A-Z0-9][^A-Z0-9]*){9}\d")
    number_pattern = re.compile(r"\d+(?:\s\d{3})*(?:,\d{1,2})?")

    def _clean_nd(value) -> str:
        text = safe_string(value)
        return "" if text.upper() == "ND" else text

    def _clean_fundusz(value) -> str:
        text = _clean_nd(value).replace("\n", " ").strip()
        text = re.sub(r"\b(Famerytura|Emerynan|Emeryvara|Eter|Emma)\b", "Emerytura", text)
        return text

    def _normalize_uniqa_fundusz(candidate: str, current: str = "") -> str:
        text = _clean_fundusz(candidate)
        if not text:
            return text

        year_match = re.search(r"(20\d{2})", text)
        if year_match:
            return f"UNIQA Emerytura {year_match.group(1)}"

        if re.fullmatch(r"UNIQA\s+Emerytura", text, flags=re.IGNORECASE):
            current_year = re.search(r"(20\d{2})", current or "")
            if current_year:
                return f"UNIQA Emerytura {current_year.group(1)}"
            return "UNIQA Emerytura 2025"

        return text

    def _clean_typ_aktywa(value) -> str:
        text = _clean_nd(value).replace("\n", " ").strip()
        lowered = text.lower()
        if re.search(r"\b(akepe|ale|maje|muye|aluye)\b", lowered):
            return "Akcje"
        if re.search(r"dłużne\s+papiery", lowered):
            return "Dłużne papiery"
        if re.search(r"\b(dhutne|dhitne|papury|papiery)\b", lowered):
            return "Obligacje"
        if re.search(r"\b(depuyty|depoz)\b", lowered):
            return "Depozyty"
        return text

    def _clean_isin(value) -> str:
        text = _clean_nd(value)
        return re.sub(r"\s+", "", text)

    def _clean_emitent(value) -> str:
        return _clean_nd(value).replace("\n", " ").strip()

    def _clean_isin_loose(value: str) -> str:
        compact = re.sub(r"[^A-Z0-9]", "", value or "")
        return compact

    def _is_valid_isin(value: str) -> bool:
        isin = safe_string(value).upper()
        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", isin):
            return False

        expanded_digits: List[str] = []
        for char in isin:
            if char.isdigit():
                expanded_digits.append(char)
            else:
                expanded_digits.extend(str(ord(char) - 55))

        digits = "".join(expanded_digits)
        total = 0
        reverse_digits = digits[::-1]
        for idx, ch in enumerate(reverse_digits):
            num = int(ch)
            if idx % 2 == 1:
                num *= 2
                if num > 9:
                    num = (num // 10) + (num % 10)
            total += num
        return total % 10 == 0

    def _extract_numbers(text: str) -> List[str]:
        numbers: List[str] = []
        for match in number_pattern.finditer(text or ""):
            if text[match.end():].lstrip().startswith("%"):
                continue
            before = text[match.start() - 1:match.start()] if match.start() > 0 else ""
            after = text[match.end():match.end() + 1]
            if (before and before.isalpha()) or (after and after.isalpha()):
                continue
            numbers.append(match.group(0))
        return numbers

    def _stitch_split_words(text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        merges = {
            ("GOSPODAR", "STWA"): "GOSPODARSTWA",
            ("DEVELOPME", "NT"): "DEVELOPMENT",
            ("KA", "SA"): "KASA",
        }
        stitched: List[str] = []
        i = 0
        while i < len(words):
            if i + 1 < len(words) and (words[i], words[i + 1]) in merges:
                stitched.append(merges[(words[i], words[i + 1])])
                i += 2
                continue
            stitched.append(words[i])
            i += 1
        return " ".join(stitched)

    def _clean_emitent_text(text: str) -> str:
        emit = safe_string(text)
        if not emit:
            return ""
        emit = re.sub(r"[A-Z]{2}[A-Z0-9]{9}\d", " ", emit)
        emit = re.sub(r"\b(?:AS?FIOUNIQAE\d|SUNIQAESFIO\d)\b", " ", emit, flags=re.IGNORECASE)
        emit = re.sub(r"\s+", " ", emit).strip(" -;,.")
        return emit

    def _is_noise_emitent(text: str) -> bool:
        emit = safe_string(text).lower()
        if emit in {"", "0", "nan"}:
            return True
        noise_patterns = [
            r"emerytur",
            r"kład\s+portf|skład\s+portf|portfel",
            r"^uniqa\b",
            r"^funduszu\b",
        ]
        return any(re.search(pattern, emit) for pattern in noise_patterns)

    rows: List[Dict[str, str]] = []
    current_fundusz = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            try:
                page_tables = page.extract_tables(table_settings)
            except Exception:
                page_tables = []
            for table in page_tables or []:
                for row in table or []:
                    if not row:
                        continue

                    row_values = [safe_string(cell) for cell in row]
                    joined = " ".join([v for v in row_values if v]).lower()
                    if "nazwa funduszu" in joined or "nazwa subfunduszu" in joined:
                        continue
                    if "skład por" in joined or "portfela dla" in joined:
                        continue

                    fundusz_candidate = ""
                    if "emerytura" in joined or "merytura" in joined:
                        year_match = re.search(r"(20\d{2})", joined)
                        if year_match:
                            fundusz_candidate = f"UNIQA Emerytura {year_match.group(1)}"
                        elif "uniqa" in joined:
                            fundusz_candidate = "UNIQA Emerytura"

                    if not fundusz_candidate:
                        for val in row_values:
                            if not val:
                                continue
                            if re.search(r"emerytura|merytura", val, re.IGNORECASE):
                                year_match = re.search(r"(20\d{2})", val)
                                if year_match:
                                    fundusz_candidate = f"UNIQA Emerytura {year_match.group(1)}"
                                elif "UNIQA" in val.upper():
                                    fundusz_candidate = val.strip()
                                else:
                                    fundusz_candidate = "UNIQA Emerytura"
                                break

                    if fundusz_candidate:
                        current_fundusz = _normalize_uniqa_fundusz(fundusz_candidate, current_fundusz)

                    if "depozyt" in joined:
                        fundusz = current_fundusz
                        if not fundusz:
                            year_match = re.search(r"(20\d{2})", joined)
                            if year_match:
                                fundusz = f"UNIQA Emerytura {year_match.group(1)}"
                                current_fundusz = fundusz

                        joined_numbers = _extract_numbers(" ".join([v for v in row_values if v]))
                        wartosc_num = parse_polish_number(joined_numbers[-1]) if joined_numbers else None
                        if wartosc_num is None:
                            continue

                        currencies = re.findall(r"\b(?:PLN|EUR|USD|GBP|CHF|JPY|CNY|SEK|NOK)\b", " ".join(row_values))
                        waluta_dep = currencies[-1] if currencies else "PLN"

                        rows.append(
                            {
                                "data": file_date,
                                "instytucja": "UNIQA TFI S.A.",
                                "fundusz": _normalize_uniqa_fundusz(fundusz, current_fundusz),
                                "typ_aktywa": "Depozyty",
                                "emitent": "Depozyty",
                                "isin": "",
                                "waluta": waluta_dep,
                                "liczba_sztuk": "",
                                "wartosc_pln": format_decimal_comma(wartosc_num),
                            }
                        )
                        continue

                    raw_isin = None
                    isin_index = None
                    for idx, value in enumerate(row_values):
                        match = isin_pattern.search(value or "")
                        if match:
                            raw_isin = match.group(0)
                            isin_index = idx
                            break
                    if not raw_isin:
                        joined_raw = " ".join([v for v in row_values if v])
                        match_loose = isin_loose_pattern.search(joined_raw)
                        if match_loose:
                            raw_isin = match_loose.group(0)
                    if not raw_isin:
                        joined_raw = " ".join([v for v in row_values if v])
                        joined_lower = joined_raw.lower()
                        is_deposit_row = any(
                            token in joined_lower
                            for token in ["depozyty", "depozyt", "środki pieniężne", "srodki pieniezne"]
                        )
                        if is_deposit_row:
                            fundusz = current_fundusz
                            if not fundusz:
                                year_match = re.search(r"(20\d{2})", joined_raw)
                                if year_match:
                                    fundusz = f"UNIQA Emerytura {year_match.group(1)}"
                                    current_fundusz = fundusz

                            joined_numbers = _extract_numbers(joined_raw)
                            wartosc_num = parse_polish_number(joined_numbers[-1]) if joined_numbers else None
                            if wartosc_num is None:
                                continue

                            currencies = re.findall(r"\b(?:PLN|EUR|USD|GBP|CHF|JPY|CNY|SEK|NOK)\b", joined_raw)
                            waluta = currencies[-1] if currencies else "PLN"

                            rows.append(
                                {
                                    "data": file_date,
                                    "instytucja": "UNIQA TFI S.A.",
                                    "fundusz": _normalize_uniqa_fundusz(fundusz, current_fundusz),
                                    "typ_aktywa": "Depozyty",
                                    "emitent": "Depozyty",
                                    "isin": "",
                                    "waluta": waluta,
                                    "liczba_sztuk": "",
                                    "wartosc_pln": format_decimal_comma(wartosc_num),
                                }
                            )
                        continue

                    isin = _clean_isin_loose(raw_isin)
                    
                    # Inteligentne mapowanie - szukaj pól po zawartości
                    fundusz = current_fundusz
                    emitent = ""
                    typ_aktywa = ""
                    waluta = ""
                    liczba_sztuk = ""
                    wartosc_pln = ""
                    
                    # Szukaj liczb - ostatnie 2 to liczba_sztuk i wartosc_pln
                    all_numbers = []
                    for idx, val in enumerate(row_values):
                        if val:
                            try:
                                float(str(val).replace(",", ".").replace(" ", ""))
                                all_numbers.append((idx, str(val)))
                            except (ValueError, TypeError):
                                pass
                    
                    # Ostatnie 2 liczby to liczba_sztuk i wartosc_pln
                    if len(all_numbers) >= 2:
                        liczba_sztuk = all_numbers[-2][1]
                        wartosc_pln = all_numbers[-1][1]
                    elif len(all_numbers) == 1:
                        wartosc_pln = all_numbers[0][1]
                    
                    # Szukaj funduszu - szukaj "merytura" i wyciągnij rok
                    for idx, val in enumerate(row_values):
                        if val and "merytura" in str(val).lower():
                            year_match = re.search(r'(\d{4})', str(val))
                            if year_match:
                                fundusz = f"UNIQA Emerytura {year_match.group(1)}"
                                break
                    
                    if not fundusz:
                        # Szukaj UNIQA
                        for val in row_values:
                            if val and "UNIQA" in str(val):
                                fundusz = str(val).strip()
                                break

                    if fundusz:
                        fundusz = _normalize_uniqa_fundusz(fundusz, current_fundusz)
                        current_fundusz = fundusz
                    
                    # Szukaj typu aktywa
                    for val in row_values:
                        if val:
                            val_str = _clean_typ_aktywa(val)
                            if not val_str:
                                continue
                            if re.match(r"^(Akcje|Obligacje|Fundusze|Fundusze inwestycyjne|Strukturyzowane|Papiery)$", val_str):
                                typ_aktywa = val_str
                                break
                            if "dłużne papiery" in val_str.lower():
                                typ_aktywa = "Dłużne papiery"
                                break

                    if not typ_aktywa:
                        for idx, val in enumerate(row_values):
                            if val and "dłużne papiery" in str(val).lower():
                                if idx + 1 < len(row_values) and "wartościowe" in str(row_values[idx + 1]).lower():
                                    typ_aktywa = "Dłużne papiery wartościowe"
                                else:
                                    typ_aktywa = "Dłużne papiery"
                                break
                                break
                    
                    # Szukaj waluty
                    for val in row_values:
                        if val:
                            val_str = str(val).strip()
                            if re.match(r"^(PLN|EUR|USD|GBP|CHF|JPY|CNY|SEK|NOK)$", val_str):
                                waluta = val_str
                                break
                    
                    # Szukaj emienta - nazwa firmy zwykle zawiera S.A., Sp., Inc. lub ma spacje
                    # Szukaj od przodu aby znaleźć długą nazwę (jeśli jest podzielona)
                    skip_patterns = {"UNIQA", "Fundusz", "SFIO", "N/D", "Specjalistyczny", "Inwestycyjny",
                                   "Otwarty", "Akcje", "Obligacje", "Fundusze", "AXA", "E", "merytura", "Emerytura"}
                    numbers_set = {num[1] for num in all_numbers}
                    
                    # Szukaj w całej kolumnie ale przywiąż się do pierwszego znalezionego
                    potential_emitents = []
                    for idx, val in enumerate(row_values):
                        if val:
                            val_str = str(val).strip()
                            # Sprawdzenie czy to potencjalnie emitent
                            if (val_str not in skip_patterns and 
                                val_str not in numbers_set and
                                not re.match(r"^[A-Z]{4}\d{3}$", val_str) and  # Nie kod AXA020
                                ("S.A." in val_str or "Sp." in val_str or " " in val_str or "INC" in val_str) and
                                len(val_str) < 60 and len(val_str) > 2 and
                                not re.match(r"^\d+$", val_str) and
                                not re.match(r"^PL|^AT|^US|^IE|^LU|^ES|^DE", val_str)):  # Nie ISIN
                                potential_emitents.append((idx, val_str))
                            
                            # Sprawdź czy to "S.A." jako oddzielny element - wtedy połącz z poprzednim
                            if val_str == "S.A." and idx > 0:
                                prev_val = row_values[idx - 1]
                                if prev_val and prev_val not in skip_patterns and prev_val not in numbers_set:
                                    combined = f"{prev_val} S.A."
                                    potential_emitents.append((idx - 1, combined))
                    
                    # Wybierz najlepszego kandydata - ten z "S.A." i najdłuższy
                    emitent = ""
                    sa_candidates = []
                    for idx, val_str in potential_emitents:
                        if val_str.endswith("S.A.") or val_str.endswith("S.A") or "S.A." in val_str:
                            sa_candidates.append(val_str)
                    
                    if sa_candidates:
                        # Wybierz najdłuższą nazwę (prawdopodobnie pełna nazwa firmy)
                        emitent = max(sa_candidates, key=len)
                    
                    # Jeśli nie znaleziono S.A., weź pierwszą długą nazwę
                    if not emitent and potential_emitents:
                        # Szukaj najdłuższej nazwy (prawdopodobnie to emitent)
                        emitent = max(potential_emitents, key=lambda x: len(x[1]))[1]

                    emitent_from_isin = ""
                    if isin_index is not None:
                        emitent_parts: List[str] = []
                        for j in range(isin_index - 1, -1, -1):
                            token = row_values[j].strip()
                            if not token:
                                if emitent_parts:
                                    break
                                continue
                            if re.match(r"^(PLN|EUR|USD|GBP|CHF|JPY|CNY|SEK|NOK)$", token):
                                break
                            if re.match(r"^\d+(?:\s\d{3})*(?:,\d{1,2})?$", token):
                                break
                            if token in skip_patterns:
                                break
                            emitent_parts.insert(0, token)
                            if len(" ".join(emitent_parts)) > 60:
                                break
                        if emitent_parts:
                            emitent_from_isin = " ".join(emitent_parts).strip()

                    if emitent_from_isin:
                        emitent = emitent_from_isin

                    if emitent:
                        emitent = _stitch_split_words(emitent)

                    if not isin_pattern.fullmatch(isin):
                        continue

                    joined_numbers = _extract_numbers(" ".join([v for v in row_values if v]))
                    liczba_num = parse_polish_number(liczba_sztuk)
                    wartosc_num = parse_polish_number(wartosc_pln)
                    if wartosc_num == 0:
                        if len(joined_numbers) >= 1:
                            wartosc_pln = joined_numbers[-1]
                            wartosc_num = parse_polish_number(wartosc_pln)
                    if liczba_num == 0 and len(joined_numbers) >= 2:
                        liczba_sztuk = joined_numbers[-2]
                        liczba_num = parse_polish_number(liczba_sztuk)

                    if not waluta:
                        currencies = re.findall(r"\b[A-Z]{3}\b", " ".join(row_values))
                        waluta = currencies[-1] if currencies else ""

                    emitent = _clean_emitent_text(emitent)

                    if not typ_aktywa:
                        emit_lower = emitent.lower()
                        if any(keyword in emit_lower for keyword in ["oblig", "bond", "skarb państwa", "treasury"]):
                            typ_aktywa = "Dłużne papiery"
                        elif any(keyword in emit_lower for keyword in ["etf", "msci", "s&p", "vanguard", "ishares", "lyxor", "sicav"]):
                            typ_aktywa = "Akcje"
                        elif emitent:
                            typ_aktywa = "Akcje"

                    if _is_noise_emitent(emitent):
                        continue

                    # Allow all rows to be written, even if one value is 0
                    # if wartosc_num == 0:
                    #     continue

                    # Oczyść liczby - usuń trailing zeros i przecinki
                    # liczba_sztuk i wartosc_pln są stringami
                    def format_number(num_str):
                        if not num_str:
                            return ""
                        # Zamień przecinek na kropkę
                        num_str = str(num_str).replace(",", ".")
                        # Konwertuj na float i z powrotem na string aby znormalizować
                        try:
                            num_val = float(num_str)
                            # Jeśli to liczba całkowita, nie dodawaj dziesiętnych
                            if num_val == int(num_val):
                                return str(int(num_val))
                            else:
                                # Usuń trailing zeros
                                return f"{num_val:.10f}".rstrip("0").rstrip(".")
                        except (ValueError, TypeError):
                            return ""
                    
                    liczba_formatted = format_number(liczba_sztuk)
                    wartosc_formatted = format_number(wartosc_pln)

                    rows.append(
                        {
                            "data": file_date,
                            "instytucja": "UNIQA TFI S.A.",
                            "fundusz": fundusz,
                            "typ_aktywa": typ_aktywa,
                            "emitent": emitent,
                            "isin": isin,
                            "waluta": waluta,
                            "liczba_sztuk": liczba_formatted,
                            "wartosc_pln": wartosc_formatted,
                        }
                    )

    df = pd.DataFrame(rows)
    if not df.empty and "fundusz" in df.columns:
        df["fundusz"] = df["fundusz"].apply(lambda x: _normalize_uniqa_fundusz(safe_string(x), ""))
    return ensure_output_schema(df)


def parse_uniqa_excel(file_path: str) -> pd.DataFrame:
    """
    Parse UNIQA TFI Excel (xlsx) files.
    Column layout (0-based):
      0  Identyfikator funduszu lub subfunduszu
      1  Nazwa funduszu
      2  Nazwa subfunduszu          -> fundusz
      3  Typ funduszu
      4  Standardowe identyfikatory subfunduszu
      5  Waluta wyceny aktywów...
      6  Nazwa emitenta             -> emitent
      7  Identyfikator instrumentu (kod ISIN) -> isin (may contain full name + ISIN)
      8  Alternatywny identyfikator
      9  Typ instrumentu            -> typ_aktywa
      10 Kategoria instrumentu
      11 Kraj emitenta
      12 Waluta instrumentu         -> waluta
      13 Ilość instrumentów         -> liczba_sztuk
      14 Wartość instrumentu        -> wartosc_pln
      15 Procentowy udział
    """
    file_date = extract_date_from_filename(file_path)
    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")

    def _nd(value) -> str:
        text = safe_string(value)
        return "" if text.upper() in ("N/D", "ND", "NAN", "NONE") else text

    rows: List[Dict[str, str]] = []
    try:
        df_raw = pd.read_excel(file_path, engine="openpyxl", header=0)
    except Exception:
        return ensure_output_schema(pd.DataFrame())

    for _, row in df_raw.iterrows():
        vals = list(row)
        if len(vals) < 15:
            continue

        fundusz_raw = _nd(vals[2]) if len(vals) > 2 else ""
        emitent_raw = _nd(vals[6]) if len(vals) > 6 else ""
        isin_raw = safe_string(vals[7]) if len(vals) > 7 else ""
        typ_aktywa_raw = _nd(vals[9]) if len(vals) > 9 else ""
        waluta_raw = _nd(vals[12]) if len(vals) > 12 else ""
        liczba_raw = _nd(vals[13]) if len(vals) > 13 else ""
        wartosc_raw = vals[14] if len(vals) > 14 else None

        # Skip rows without a value
        if wartosc_raw is None:
            continue

        # Normalize isin - sometimes the col contains full fund name + ISIN at end
        isin = ""
        if isin_raw.upper() not in ("N/D", "ND", ""):
            match = isin_pattern.search(isin_raw)
            if match:
                isin = match.group(0)
                # If emitent is empty and there's text before the ISIN, use it
                prefix = isin_raw[: match.start()].strip(" .-")
                if not emitent_raw and prefix:
                    emitent_raw = prefix
            else:
                # Non-standard identifier, keep as-is
                isin = isin_raw.strip()

        # Normalize N/D liczba_sztuk
        liczba = liczba_raw if liczba_raw.upper() not in ("N/D", "ND") else ""

        rows.append(
            {
                "data": file_date,
                "instytucja": "UNIQA TFI S.A.",
                "fundusz": fundusz_raw,
                "typ_aktywa": typ_aktywa_raw,
                "emitent": emitent_raw,
                "isin": isin,
                "waluta": waluta_raw,
                "liczba_sztuk": liczba,
                "wartosc_pln": format_decimal_comma(wartosc_raw),
            }
        )

    return ensure_output_schema(pd.DataFrame(rows))


def parse_nn_pdf(file_path: str) -> pd.DataFrame:
    """
    Parse Nationale-Nederlanden (NN) PDF files.
        File formats:
            1. NN_[ROK_PPK]_[DATA_RAPORTU].pdf (e.g., NN_25_2025-12-31.pdf)
            2. [DATA]_struktura_aktywow_roczna_NN_ DFE_NJ_[ROK].pdf (e.g., 20251231_struktura_aktywow_roczna_NN_ DFE_NJ_30.pdf)
    """
    rows: List[Dict[str, str]] = []
    
    # Extract date and fund year from filename
    filename = os.path.basename(file_path)
    
    # Try pattern 1: NN_YY_YYYY-MM-DD.pdf (with optional space after NN_)
    match = re.search(r'NN_\s*(\d{2})_(\d{4}-\d{2}-\d{2})\.pdf', filename, re.IGNORECASE)
    if match:
        year_suffix = match.group(1)  # "25", "30", etc.
        file_date = match.group(2)    # "2025-12-31"
    else:
        # Try pattern 2: YYYYMMDD_struktura_aktywow_roczna_NN_ DFE_NJ_YY.pdf
        match = re.search(r'(\d{8})_struktura_aktywow.*_NJ_(\d{2})\.pdf', filename, re.IGNORECASE)
        if match:
            date_str = match.group(1)  # "20251231"
            year_suffix = match.group(2)  # "30", "35", etc.
            # Convert YYYYMMDD to YYYY-MM-DD
            file_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            return ensure_output_schema(pd.DataFrame())
    
    fundusz_name = f"NN DFE Nasze Jutro 20{year_suffix}"
    
    # Read PDF tables
    try:
        with pdfplumber.open(file_path) as pdf:
            current_section = None
            
            for page in pdf.pages:
                tables = page.extract_tables()
                
                for table in tables or []:
                    for row in table or []:
                        if not row or len(row) < 4:
                            continue
                        
                        lp = safe_string(row[0])
                        kategoria = safe_string(row[1])
                        udzial = safe_string(row[2])
                        wartosc = safe_string(row[3])
                        
                        # Skip header row
                        if "Kategoria lokaty" in kategoria or "Lp." in lp:
                            continue
                        
                        # Detect section headers (numbered categories)
                        # Format: "1" or "25" in Lp column, long description in kategoria
                        if lp and lp.isdigit() and len(kategoria) > 50:
                            current_section = lp
                            continue  # Skip summary rows
                        
                        # Process detail rows (empty Lp, specific instrument name)
                        if not lp and kategoria and wartosc:
                            if not current_section:
                                continue
                            
                            # Determine typ_aktywa based on section number
                            typ_aktywa = ""
                            section_num = int(current_section) if current_section else 0
                            
                            if section_num in [1, 3]:
                                typ_aktywa = "Obligacje skarbowe/gwarantowane"
                            elif section_num in [5, 6, 36]:
                                typ_aktywa = "Depozyt/Gotówka"
                            elif section_num == 25:
                                typ_aktywa = "Obligacje korporacyjne"
                            elif 7 <= section_num <= 14:
                                typ_aktywa = "Akcje"
                            else:
                                typ_aktywa = "Inne"
                            
                            # Extract emitent and ISIN from kategoria
                            # Format: "MINISTERSTWO FINANSÓW - DS1033 - 25/10/2033"
                            # or: "Alior Bank S.A. - ALR1029 - 19/10/2029"
                            emitent = kategoria
                            isin = ""
                            
                            # Try to extract code (e.g., DS1033, ALR1029, FPC0328)
                            code_match = re.search(r'\s-\s([A-Z]{2,3}\d{4})\s-\s', kategoria)
                            if code_match:
                                isin = code_match.group(1)
                                # Extract emitent (everything before first " - ")
                                emitent = kategoria.split(' - ')[0].strip()
                            else:
                                # No code found, clean emitent
                                # Remove date patterns like "25/04/2032"
                                emitent = re.sub(r'\s-\s\d{2}/\d{2}/\d{4}$', '', kategoria).strip()
                            
                            # Clean value (remove spaces)
                            wartosc_clean = wartosc.replace(' ', '').replace(',', '.')
                            
                            rows.append({
                                "data": file_date,
                                "instytucja": "Nationale-Nederlanden",
                                "fundusz": fundusz_name,
                                "typ_aktywa": typ_aktywa,
                                "emitent": emitent,
                                "isin": isin,
                                "waluta": "PLN",
                                "liczba_sztuk": "0",
                                "wartosc_pln": wartosc_clean,
                            })
    
    except Exception:
        return ensure_output_schema(pd.DataFrame())
    
    df = pd.DataFrame(rows)
    return ensure_output_schema(df)


def parse_pocztylion_pdf(file_path: str) -> pd.DataFrame:
    instytucja = "Pocztylion"

    file_date = extract_date_from_filename(file_path)
    rows: List[Dict[str, str]] = []

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        page_texts = [(page.extract_text() or "") for page in reader.pages]

    full_text = "\n".join(page_texts)
    if not file_date:
        date_match = re.search(r"Data\s+wyceny\s*:\s*(\d{2})[./](\d{2})[./](\d{4})", full_text, flags=re.IGNORECASE)
        if date_match:
            day, month, year = date_match.groups()
            file_date = f"{year}-{month}-{day}"

    fundusz = ""
    fund_match = re.search(r"PPK\s+Pocztylion\s+\d{4}\s+DFE", full_text, flags=re.IGNORECASE)
    if fund_match:
        fundusz = re.sub(r"\s+", " ", fund_match.group(0)).strip()

    category_pattern = re.compile(r"^\s*\d+\.\s+[A-ZĄĆĘŁŃÓŚŹŻ].*$")
    trailing_totals_pattern = re.compile(r"\s+\d{1,3}(?:\s\d{3})*,\d{2}\s+\d{1,3}(?:\s\d{3})*,\d{2}%\s*$")
    asset_pattern = re.compile(r"^\s*(.+?)\s+([\d\s]+,\d{2})\s+([\d\s]+,\d{2})%\s*$")
    continuation_prefixes = (
        "przez ",
        "także ",
        "takze ",
        "a także ",
        "a takze ",
        "oraz ",
        "w tym ",
        "łącznie ",
        "lacznie ",
        "zawarte ",
        "których ",
        "ktorych ",
        "prawach do akcji",
        "inwestycyjne ",
        "gwarantowane ",
        "oprocentowaniu",
        "pkt ",
        "ust.",
        "9a.",
    )

    current_category = ""

    def _category_to_std(category: str) -> str:
        return normalize_typ_aktywa(category)

    for page_text in page_texts:
        for raw_line in page_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue

            if category_pattern.match(line):
                category_clean = trailing_totals_pattern.sub("", line).strip()
                current_category = category_clean
                continue

            match = asset_pattern.match(line)
            if not match:
                continue

            aktywo, wartosc_raw, _udzial = match.groups()
            aktywo = re.sub(r"\s+", " ", aktywo).strip()
            if not aktywo:
                continue

            aktywo_lower = aktywo.lower()
            if aktywo_lower.startswith("razem") or aktywo_lower.startswith("suma"):
                continue
            if re.fullmatch(r"\d+(?:[.,]\d+)?", aktywo):
                continue
            if re.match(r"^\d+\.\s+", aktywo):
                continue
            if ";" in aktywo:
                continue
            if aktywo_lower.startswith(continuation_prefixes):
                continue
            if current_category and aktywo_lower == current_category.lower():
                continue
            if current_category:
                current_category_lower = current_category.lower()
                if aktywo_lower in current_category_lower or current_category_lower in aktywo_lower:
                    continue
                if _normalize_equity_name(aktywo) == _normalize_equity_name(current_category):
                    continue

            if len(aktywo.split()) > 8 and any(
                token in aktywo_lower
                for token in [
                    "obligacje",
                    "papiery wartościowe",
                    "papiery wartosciowe",
                    "pożyczki",
                    "pozyczki",
                    "kredyty",
                    "udzielane",
                    "podmiotom",
                    "emitowane",
                    "środki pieniężne",
                    "srodki pieniezne",
                    "tytuły uczestnictwa",
                    "tytuly uczestnictwa",
                ]
            ):
                continue

            if aktywo_lower.startswith(("emitowane przez", "a także", "a takze", "oraz ", "przez ")):
                continue

            wartosc_num = parse_polish_number(wartosc_raw)
            if wartosc_num is None:
                continue
            if is_timestamp_like_amount(wartosc_raw):
                continue

            row = {
                "data": file_date,
                "instytucja": instytucja,
                "fundusz": fundusz,
                "typ_aktywa": current_category,
                "emitent": aktywo,
                "isin": "",
                "waluta": "PLN",
                "liczba_sztuk": "",
                "wartosc_pln": format_decimal_comma(wartosc_num),
                "TYP_aktywo_std": _category_to_std(current_category),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return ensure_output_schema(df)

    df = ensure_output_schema(df)
    if fundusz:
        df["fundusz"] = df["fundusz"].replace("nan", fundusz)
    if file_date:
        df["data"] = df["data"].replace("nan", file_date)

    df["DATA_fundusz"] = (
        df["fundusz"]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .fillna("nan")
    )

    if "typ_aktywa" in df.columns:
        df["TYP_aktywo_std"] = df["typ_aktywa"].apply(normalize_typ_aktywa)

    return df


def parse_generali_excel(file_path: str) -> pd.DataFrame:
    """
    Parse Generali Excel file.
    Excel format: Arkusz 'Horyzont' z 16 kolumnami
    Mapowanie:
    - Nazwa funduszu / subfunduszu -> fundusz
    - Identyfikator instrumentu (kod ISIN) -> isin
    - Nazwa emitenta -> emitent
    - Typ instrumentu -> typ_aktywa
    - Waluta instrumentu -> waluta
    - Ilość instrumentów w portfelu -> liczba_sztuk
    - Wartość instrumentu w walucie wyceny funduszu -> wartosc_pln
    """
    instytucja = "Generali Investments TFI S.A."
    
    # Extract date from filename (format: Generali_YYYY-MM-DD.xlsx)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(file_path))
    if date_match:
        data = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    else:
        data = ""
    
    rows: List[Dict[str, str]] = []
    
    try:
        # Wczytaj Excel - szukaj arkusza zawierającego dane portfela
        xls = pd.ExcelFile(file_path)
        sheet_name = None
        
        # Szukaj arkusza z danymi (może być "Horyzont", "Portfel", itp.)
        for name in xls.sheet_names:
            if name.strip().lower() in ["horyzont", "portfel", "portfolio"]:
                sheet_name = name
                break
        
        if not sheet_name:
            sheet_name = xls.sheet_names[0]
        
        # Spróbuj czytać z prawidłowym nagłówkiem (może być w wierszu 1)
        # Czytaj bez nagłówka najpierw, aby sprawdzić strukturę
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=3)
        
        # Sprawdź gdzie jest nagłówek (zwykle wiersz zawierający "Identyfikator")
        header_row = 0
        for idx, val in enumerate(df_raw[0]):
            if str(val).lower().startswith("identyfikator"):
                header_row = idx
                break
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        
        if df.empty:
            return ensure_output_schema(df)
        
        # Mapowanie kolumn
        # Znajdź kolumny - nazwy mogą się różnić między wersjami
        def _find_col(keywords: List[str]) -> Optional[str]:
            for col in df.columns:
                col_norm = normalize_header(col)
                if any(key in col_norm for key in keywords):
                    return col
            return None

        fund_col = _find_col([
            "nazwa subfunduszu",
            "nazwa funduszu / subfunduszu",
            "nazwa funduszu",
        ])
        isin_col = _find_col(["kod isin", "identyfikator instrumentu (kod isin)"])
        emitent_col = _find_col(["nazwa emitenta", "emitent"])
        typ_col = _find_col(["typ instrumentu", "rodzaj instrumentu"])
        waluta_col = _find_col(["waluta instrumentu", "waluta notowań"])
        liczba_col = _find_col(["ilość instrumentów", "ilość"])

        wartosc_col = None
        for col in df.columns:
            col_norm = normalize_header(col)
            if "wartość instrumentu" in col_norm and ("wyceny" in col_norm or "walucie" in col_norm):
                wartosc_col = col
                break
        if not wartosc_col:
            wartosc_col = _find_col(["wartość całkowita", "wartość"])
        
        for idx, row in df.iterrows():
            # Pobierz dane z kolumn
            
            fundusz = safe_string(row.get(fund_col, "") if fund_col else "").strip()
            isin = safe_string(row.get(isin_col, "") if isin_col else "").strip()
            emitent = safe_string(row.get(emitent_col, "") if emitent_col else "").strip()
            typ_aktywa = safe_string(row.get(typ_col, "") if typ_col else "").strip()
            waluta = safe_string(row.get(waluta_col, "") if waluta_col else "").strip()
            # Przeskocz wiersze bez istotnych danych
            if not isin or isin == "N/D":
                continue
            
            # Czyszczenie danych
            # Usuń "N/D" z pól tekstowych
            if fundusz == "N/D":
                fundusz = ""
            if emitent == "N/D":
                emitent = ""
            if typ_aktywa == "N/D":
                typ_aktywa = ""
            
            # Parsuj liczby
            liczba_sztuk_raw = row.get(liczba_col, "")
            wartosc_pln_raw = row.get(wartosc_col, "")
            
            # Konwertuj do string dla parsowania
            liczba_sztuk = safe_string(parse_polish_number(liczba_sztuk_raw)).strip()
            wartosc_pln = format_decimal_comma(parse_polish_number(wartosc_pln_raw))
            
            # Przeskocz wiersze bez ISIN
            if not isin:
                continue
            
            rows.append({
                "data": data,
                "instytucja": instytucja,
                "fundusz": fundusz,
                "typ_aktywa": typ_aktywa,
                "emitent": emitent,
                "isin": isin,
                "waluta": waluta,
                "liczba_sztuk": liczba_sztuk,
                "wartosc_pln": wartosc_pln,
            })
    
    except Exception:
        pass
    
    df = pd.DataFrame(rows)
    return ensure_output_schema(df)


def parse_vienna_pdf(file_path: str) -> pd.DataFrame:
    instytucja = "Vienna Life"
    file_date = extract_date_from_filename(file_path)
    if not file_date:
        preview_text = extract_text_pdfminer(file_path, page_numbers=[0], timeout_seconds=8)
        if not preview_text:
            preview_text = extract_text_from_pdf(file_path, max_pages=1)
        file_date = parse_date_from_text(preview_text or "") or ""

    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")
    number_pattern = re.compile(r"-?\d{1,3}(?:\s\d{3})*(?:,\d+)?|-?\d+(?:,\d+)?")
    currency_pattern = re.compile(r"^[A-Z]{3}$")

    rows: List[Dict[str, str]] = []

    def _extract_fundusz(line: str) -> str:
        match = re.search(r"\b(UFK\s+.+?\s20\d{2})\b", line)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _extract_numbers_after_currency(tokens: List[str], start_idx: int) -> List[str]:
        tail = " ".join(tokens[start_idx + 1 :])
        tail = re.sub(r"\s+", " ", tail).strip()
        if not tail:
            return []
        numbers: List[str] = []
        for match in number_pattern.finditer(tail):
            suffix = tail[match.end():].lstrip()
            if suffix.startswith("%"):
                continue
            numbers.append(match.group(0).strip())
        return numbers

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if not page_text:
                continue

            for raw_line in page_text.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line or "DOBRA PRAKTYKA" in line.upper():
                    continue
                if "%" not in line:
                    continue

                isin_match = isin_pattern.search(line)
                if not isin_match:
                    continue
                isin = isin_match.group(0)

                tokens = line.split()
                isin_idx = next((i for i, token in enumerate(tokens) if token == isin), -1)
                if isin_idx < 0:
                    continue

                fundusz = _extract_fundusz(line)

                emitent_tokens = tokens[:isin_idx]
                if fundusz:
                    fund_parts = fundusz.split()
                    try:
                        start_idx = next(i for i, token in enumerate(emitent_tokens) if token.upper() == "UFK")
                        end_idx = start_idx + len(fund_parts)
                        pre = emitent_tokens[:start_idx]
                        post = emitent_tokens[end_idx:]
                        emitent_tokens = pre + post
                    except StopIteration:
                        pass

                while emitent_tokens:
                    token_upper = emitent_tokens[0].upper()
                    if re.fullmatch(r"Y_PPK\d{2}_A", token_upper) or token_upper in {"PPK", "N/D"}:
                        emitent_tokens = emitent_tokens[1:]
                        continue
                    break
                while emitent_tokens and currency_pattern.match(emitent_tokens[0].upper()):
                    emitent_tokens = emitent_tokens[1:]
                emitent = " ".join(emitent_tokens).strip()

                post_tokens = tokens[isin_idx + 1 :]
                if not post_tokens:
                    continue

                nd_idx = next((i for i, token in enumerate(post_tokens) if token.upper() == "N/D"), -1)
                if nd_idx <= 0:
                    continue

                typ_aktywa = " ".join(post_tokens[:nd_idx]).strip()

                waluta = ""
                if nd_idx + 2 < len(post_tokens) and currency_pattern.match(post_tokens[nd_idx + 2].upper()):
                    waluta = post_tokens[nd_idx + 2].upper()
                elif nd_idx + 1 < len(post_tokens) and currency_pattern.match(post_tokens[nd_idx + 1].upper()):
                    waluta = post_tokens[nd_idx + 1].upper()

                currency_idx = -1
                if waluta:
                    for i in range(len(post_tokens) - 1, -1, -1):
                        if post_tokens[i].upper() == waluta:
                            currency_idx = i
                            break
                if currency_idx < 0:
                    continue

                numbers = _extract_numbers_after_currency(post_tokens, currency_idx)
                if not numbers:
                    continue

                liczba_sztuk = numbers[0]
                if re.search(r"obligacje", typ_aktywa, flags=re.IGNORECASE) and len(numbers) >= 2:
                    wartosc_pln = numbers[-2]
                else:
                    wartosc_pln = numbers[-1]

                rows.append(
                    {
                        "data": file_date,
                        "instytucja": instytucja,
                        "fundusz": fundusz,
                        "typ_aktywa": typ_aktywa,
                        "emitent": emitent,
                        "isin": isin,
                        "waluta": waluta,
                        "liczba_sztuk": liczba_sztuk,
                        "wartosc_pln": wartosc_pln,
                    }
                )

    if not rows:
        return ensure_output_schema(pd.DataFrame())

    df = pd.DataFrame(rows)

    if not df.empty:
        qty_num = df["liczba_sztuk"].apply(parse_polish_number)
        value_num = df["wartosc_pln"].apply(parse_polish_number)
        akcje_mask = df["typ_aktywa"].astype(str).str.contains(r"akcje", case=False, na=False)

        group_keys = ["data", "fundusz", "isin", "waluta"]
        has_positive_qty = (
            (qty_num.fillna(0) > 0)
            .groupby([df[key] for key in group_keys], dropna=False)
            .transform("max")
            .astype(bool)
        )

        dust_duplicate_mask = akcje_mask & qty_num.fillna(0).eq(0) & value_num.fillna(0).gt(0) & has_positive_qty
        if dust_duplicate_mask.any():
            df = df.loc[~dust_duplicate_mask].copy()

    schema_df = ensure_output_schema(df)

    schema_df["fundusz"] = schema_df["fundusz"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    schema_df["DATA_fundusz"] = (
        schema_df["fundusz"]
        .astype(str)
        .str.extract(r"(20\d{2})\s*$", expand=False)
        .fillna("nan")
    )

    typ_series = schema_df["typ_aktywa"].astype(str)
    schema_df["TYP_aktywo_std"] = typ_series.apply(normalize_typ_aktywa)

    meaningful_mask = (
        schema_df["fundusz"].astype(str).str.strip().str.lower().ne("nan")
        | schema_df["isin"].astype(str).str.strip().str.lower().ne("nan")
        | schema_df["emitent"].astype(str).str.strip().str.lower().ne("nan")
        | schema_df["wartosc_pln"].astype(str).str.strip().str.lower().ne("nan")
    )
    schema_df = schema_df[meaningful_mask].copy()

    return schema_df


# -------------------------
# Detection & main flow
# -------------------------

def detect_parser(file_path: str) -> Optional[str]:
    name = os.path.basename(file_path).lower()
    if name.startswith("pzu2_") and re.search(r"\d{4}-\d{2}-\d{2}", name) and name.endswith((".xls", ".xlsx")):
        return "pzu2"
    if "pko" in name and ("emerytura" in name or "sklad_portfela" in name) and name.endswith((".xls", ".xlsx", ".csv")):
        return "pko"
    if "pfr" in name and "sklad_portfela" in name and name.endswith((".xls", ".xlsx", ".csv")):
        return "pfr"
    if "allianz" in name and name.endswith((".xls", ".xlsx")):
        return "allianz"
    if "santander" in name and name.endswith((".xls", ".xlsx")):
        return "santander"
    if name.startswith("bnp_") and re.search(r"\d{4}-\d{2}-\d{2}", name) and name.endswith((".xls", ".xlsx")):
        return "bnp"
    if name.startswith("erste_") and re.search(r"\d{4}-\d{2}-\d{2}", name) and name.endswith((".xls", ".xlsx")):
        return "erste"
    if name.startswith("ing_") and re.search(r"\d{4}-\d{2}-\d{2}", name) and name.endswith((".xls", ".xlsx")):
        return "ing"
    if "goldman" in name and name.endswith((".xls", ".xlsx")):
        return "goldman"
    # Millennium - handle both "Millennium" (2 l's) and "Millenium" (1 l) spellings
    if "millen" in name and name.endswith((".xls", ".xlsx")):
        return "millennium"
    if name.endswith((".xls", ".xlsx")) and "skład portfela" in name:
        if is_millennium_excel(file_path):
            return "millennium"
    
    # PZU detection: filename format [instytucja]_YYYY-MM-DD.xlsx
    if "pzu" in name and re.search(r"\d{4}-\d{2}-\d{2}", name) and name.endswith((".xls", ".xlsx")):
        return "pzu"

    # Fallback: Check if it's a PKO file by examining headers
    if "pko" in name and name.endswith((".xls", ".xlsx", ".csv")):
        try:
            df = pd.read_excel(file_path, engine="openpyxl", header=0, nrows=1) if name.endswith((".xls", ".xlsx")) else pd.read_csv(file_path, header=0, nrows=1)
            headers_normalized = [normalize_header(c) for c in df.columns]
            if any("identyfikator instrumentu" in h for h in headers_normalized) and any("wartosc instrumentu" in h for h in headers_normalized):
                return "pko"
        except Exception:
            pass

    # Fallback: Check if it's a PFR file by examining headers
    if "pfr" in name and name.endswith((".xls", ".xlsx", ".csv")):
        try:
            df = pd.read_excel(file_path, engine="openpyxl", header=0, nrows=1) if name.endswith((".xls", ".xlsx")) else pd.read_csv(file_path, header=0, nrows=1)
            headers_normalized = [normalize_header(c) for c in df.columns]
            if any("kod isin" in h for h in headers_normalized) and any("wartość" in h for h in headers_normalized):
                return "pfr"
        except Exception:
            pass

    if "uniqa" in name and name.endswith((".xlsx", ".xls")):
        return "uniqa_excel"
    if "uniqa" in name and name.endswith(".pdf"):
        return "uniqa"
    if re.match(r"^vienna_\d{4}-\d{2}-\d{2}\.pdf$", name):
        return "vienna"
    if re.match(r"^pocztylion_[a-z0-9]{2}_\d{4}-\d{2}-\d{2}\.pdf$", name):
        return "pocztylion"
    
    # Generali: Excel format
    if "generali" in name and name.endswith((".xlsx", ".xls")):
        return "generali"

    if name.endswith(".txt"):
        if "esalian" in name or "esalien" in name:
            return "esaliens_text"

    if name.endswith(".txt"):
        if "esalian" in name or "esalien" in name:
            return "esaliens_txt"

    if name.endswith(('.xlsx', '.xls')):
        if "pekao" in name:
            return "pekao_excel"

    if name.endswith(".pdf"):
        if "investors" in name or "investor" in name:
            return "investors"
        if "pekao" in name:
            return "pekao"
        # Esaliens - handle both "Esaliens" and "Esalians" spellings
        if "esalian" in name or "esalien" in name:
            return "esaliens"
        if "horyzont" in name and "sfio" in name:
            # Old Generali PDF - skip, use Excel instead
            return None
        if (name.startswith("nn_") and re.search(r'nn_\s*\d{2}_\d{4}-\d{2}-\d{2}\.pdf', name)) or \
           ("struktura_aktywow" in name and "dfe_nj" in name and re.search(r'_nj_\d{2}\.pdf', name)):
            return "nn"

        try:
            text = extract_text_pdfminer(file_path, page_numbers=[0], timeout_seconds=8).lower()
        except Exception:
            text = ""
        if "nazwa funduszu / subfunduszu" in text or "generali" in text:
            return "generali"
        if "nazwa funduszu" in text or "esalian" in text or "esalien" in text:
            return "esaliens"

    return None


def process_folder(folder_path: str) -> pd.DataFrame:
    quarter_end = quarter_end_date_from_folder(os.path.basename(folder_path)) or ""
    folder_name = os.path.basename(folder_path).lower()

    parser_map: Dict[str, Callable[[str], pd.DataFrame]] = {
        "allianz": parse_allianz_excel,
        "santander": parse_santander_excel,
        "bnp": parse_bnp_excel,
        "goldman": parse_goldman_excel,
        "millennium": parse_millennium_excel,
        "pfr": parse_pfr_excel,
        "pko": lambda p: parse_pko_excel(p, quarter_end),
        "pzu": parse_pzu_excel,
        "pzu1": parse_pzu1_excel,
        "pzu2": parse_pzu2_excel,
        "erste": parse_erste_excel,
        "ing": parse_ing_excel,
        "esaliens": parse_esaliens_pdf,
        "esaliens_txt": parse_esaliens_text,
        "esaliens_text": parse_esaliens_text_file,
        "generali": parse_generali_excel,
        "uniqa": parse_uniqa_pdf,
        "uniqa_excel": parse_uniqa_excel,
        "vienna": parse_vienna_pdf,
        "pocztylion": parse_pocztylion_pdf,
        "nn": parse_nn_pdf,
        "investors": parse_investors_pdf,
        "pekao": parse_pekao_pdf,
        "pekao_excel": parse_pekao_excel,
        # TU DODAĆ KOLEJNE
    }

    def run_parser_with_timeout(
        parser_fn: Callable[[str], pd.DataFrame],
        path: str,
        timeout_seconds: int = 20,
    ) -> pd.DataFrame:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()

        def _worker(queue: mp.Queue) -> None:
            try:
                queue.put(parser_fn(path))
            except Exception as exc:  # pragma: no cover
                queue.put(exc)

        process = ctx.Process(target=_worker, args=(result_queue,))
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return ensure_output_schema(pd.DataFrame())

        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                return ensure_output_schema(pd.DataFrame())
            return result

        return ensure_output_schema(pd.DataFrame())

    all_rows: List[pd.DataFrame] = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        file_name = os.path.basename(file_path).lower()
        if (
            folder_name == "raw_4q22"
            and file_name.startswith("pzu2_")
            and file_name.endswith((".xlsx", ".xls"))
        ):
            parser_key = "pzu2"
        elif (
            folder_name == "raw_4q22"
            and file_name.startswith("pzu1_")
            and file_name.endswith((".xlsx", ".xls"))
        ):
            parser_key = "pzu1"
        elif (
            folder_name in {"raw_4q23", "raw_4q24"}
            and file_name.startswith("pzu1_")
            and file_name.endswith((".xlsx", ".xls"))
        ):
            parser_key = "pzu1"
        else:
            parser_key = detect_parser(file_path)
        if not parser_key:
            continue
        parser = parser_map.get(parser_key)
        if not parser:
            continue
        if file_path.lower().endswith(".pdf"):
            if parser_key in ("generali", "uniqa", "vienna", "pocztylion", "investors", "pekao", "esaliens", "nn"):
                df = parser(file_path)
            else:
                df = run_parser_with_timeout(parser, file_path, timeout_seconds=20)
        else:
            df = parser(file_path)
        df = fill_missing_date_from_title(df, file_path)
        all_rows.append(df)

    if not all_rows:
        return ensure_output_schema(pd.DataFrame())

    return pd.concat(all_rows, ignore_index=True)


def validate_knf_reconciliation(master_df: pd.DataFrame, base_dir: str, threshold_pct: float = 5.0) -> None:
    knf_path = os.path.join(base_dir, "clear", "knf_reference.csv")
    if not os.path.exists(knf_path) or master_df.empty:
        return

    knf_df = pd.read_csv(knf_path, sep=";", dtype=str)
    required_knf_cols = {"instytucja", "4Q23_knf", "4Q24_knf", "4Q25_knf"}
    if not required_knf_cols.issubset(knf_df.columns):
        return

    work = master_df.copy()
    work["wartosc_num"] = work.get("wartosc_pln", "").apply(parse_polish_number)
    work["quarter"] = work.get("data", "").astype(str).str.extract(r"^(\d{4})-(\d{2})-")[0].str[-2:].radd("4Q")

    inst_map = {
        "ESALIENS TFI S.A.": "Esaliens TFI S.A.",
        "Millennium TFI S.A.": "MILLENNIUM TFI S.A.",
        "Nationale-Nederlanden": "Nationale-Nederlanden PTE S.A.",
        "PZU TFI S.A.": "TFI PZU SA",
        "Vienna Life": "Vienna",
    }
    work["instytucja"] = work.get("instytucja", "").astype(str).str.strip().replace(inst_map)
    work = work[work["quarter"].isin(["4Q23", "4Q24", "4Q25"])]

    master_agg = (
        work.groupby(["instytucja", "quarter"], dropna=False)["wartosc_num"]
        .sum()
        .reset_index()
        .rename(columns={"wartosc_num": "wartosc_master"})
    )

    knf_long = knf_df.melt(
        id_vars=["instytucja"],
        value_vars=["4Q23_knf", "4Q24_knf", "4Q25_knf"],
        var_name="q",
        value_name="wartosc_knf",
    )
    knf_long["quarter"] = knf_long["q"].str.replace("_knf", "", regex=False)
    knf_long["wartosc_knf"] = knf_long["wartosc_knf"].apply(parse_polish_number)

    report = knf_long[["instytucja", "quarter", "wartosc_knf"]].merge(
        master_agg,
        on=["instytucja", "quarter"],
        how="left",
    )
    report["wartosc_master"] = report["wartosc_master"].fillna(0)
    report["ratio_pct"] = report.apply(
        lambda row: (row["wartosc_master"] / row["wartosc_knf"] * 100)
        if row["wartosc_knf"] not in (0, None) and not pd.isna(row["wartosc_knf"])
        else pd.NA,
        axis=1,
    )
    report["abs_dev_pct"] = (report["ratio_pct"] - 100).abs()
    report["diff_pln"] = report["wartosc_master"] - report["wartosc_knf"]
    report["status"] = report["abs_dev_pct"].apply(
        lambda x: "OK" if pd.notna(x) and x <= threshold_pct else "ALERT"
    )

    report_out = os.path.join(base_dir, "output_csv", "knf_reconciliation_report.csv")
    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    report.to_csv(report_out, sep=";", index=False, encoding="utf-8-sig")

    alerts = report[(report["status"] == "ALERT") & report["wartosc_knf"].notna()]
    if not alerts.empty:
        top = alerts.sort_values("abs_dev_pct", ascending=False).head(8)
        summary = ", ".join(
            f"{row.instytucja} {row.quarter}: {row.ratio_pct:.2f}%"
            for _, row in top.iterrows()
            if pd.notna(row.ratio_pct)
        )
        raise RuntimeError(
            f"KNF validation failed (> {threshold_pct}%): {len(alerts)} alertów. Szczegóły: {summary}. "
            f"Pełny raport: {report_out}"
        )


def main() -> None:
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output_csv")
    os.makedirs(output_dir, exist_ok=True)
    equity_map, equity_name_map = load_equity_mapping(base_dir)
    isin_map = load_isin_mapping(base_dir)
    manual_shares_map = load_manual_shares_map(base_dir)
    cleanup_percent_outputs(output_dir)

    raw_folders = [
        path
        for path in glob.glob(os.path.join(base_dir, "raw_*"))
        if os.path.isdir(path)
    ]

    holdings_by_quarter: Dict[str, pd.DataFrame] = {}
    fund_positions_by_quarter: Dict[str, pd.DataFrame] = {}
    source_rows_by_quarter: Dict[str, pd.DataFrame] = {}

    for folder in raw_folders:
        quarter_token = quarter_token_from_folder(os.path.basename(folder)) or ""
        if not quarter_token:
            continue
        result_df = process_folder(folder)
        output_path = os.path.join(output_dir, f"PPK_{quarter_token}.csv")
        result_df = result_df.copy()
        result_df = apply_manual_shares_map(result_df, quarter_token, manual_shares_map)
        for col in ["liczba_sztuk", "wartosc_pln"]:
            if col in result_df.columns:
                result_df[col] = result_df[col].apply(format_decimal_comma)
        result_df = apply_equity_nazwa(result_df, equity_map, equity_name_map)
        result_df = fill_missing_isin(result_df, isin_map)
        result_df, removed_anomalies = sanitize_wartosc_pln_anomalies(result_df)
        if removed_anomalies:
            print(f"Usunięto anomalie wartosc_pln ({quarter_token}): {removed_anomalies}")
        fund_share_df = build_equity_share_pivot(result_df)
        holdings_by_quarter[quarter_token] = build_equity_holdings_numeric(result_df)
        fund_positions_by_quarter[quarter_token] = build_fund_position_share(result_df)
        source_rows_by_quarter[quarter_token] = result_df.copy()
        result_df.to_csv(
            output_path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )
        fund_share_path = os.path.join(output_dir, f"PPK_{quarter_token}_holdings_pct.csv")
        fund_share_df.to_csv(
            fund_share_path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    # Generate YoY change files for all quarters where previous-year quarter exists
    for quarter_token in sorted(holdings_by_quarter.keys(), key=quarter_sort_key):
        prev_token = prev_quarter_token(quarter_token)
        if not prev_token:
            continue
        prev_df = holdings_by_quarter.get(prev_token, pd.DataFrame())
        curr_df = holdings_by_quarter.get(quarter_token, pd.DataFrame())
        if prev_df.empty and curr_df.empty:
            continue
        change_df = build_change_table(prev_df, curr_df)
        # name like PPK_23-24_chg.csv (use two-digit years)
        out_name = f"PPK_{prev_token[-2:]}-{quarter_token[-2:]}_chg.csv"
        change_path = os.path.join(output_dir, out_name)
        change_df.to_csv(change_path, index=False, sep=";", encoding="utf-8-sig")

    master_df = build_master_dataset(source_rows_by_quarter)
    if not master_df.empty:
        master_path = os.path.join(output_dir, "PPK_master.csv")
        master_df.to_csv(
            master_path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
