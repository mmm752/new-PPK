import os
import glob
import re
import signal
import multiprocessing as mp
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Dict, List, Optional

import pandas as pd
import pdfplumber
import PyPDF2
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover
    pdfminer_extract_text = None

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
    "B_ticker",
]

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


def format_decimal_comma(value) -> str:
    if value is None or value == "":
        return "nan"
    if isinstance(value, float) and pd.isna(value):
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

    if re.search(r"\bakcj", text):
        return "akcje"
    if re.search(r"\boblig", text) or "dłużn" in text or "dluzn" in text:
        return "obligacje"
    if "list zastawn" in text or "papier komerc" in text:
        return "obligacje"
    if "instrument" in text and "pochodn" in text:
        return "inne"

    return "inne"


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


def quarter_end_date_from_folder(folder_name: str) -> Optional[str]:
    match = re.search(r"raw_(\d)q(\d{2})", folder_name, re.IGNORECASE)
    if not match:
        return None
    quarter = int(match.group(1))
    year = 2000 + int(match.group(2))
    if quarter == 1:
        return f"{year}-03-31"
    if quarter == 2:
        return f"{year}-06-30"
    if quarter == 3:
        return f"{year}-09-30"
    if quarter == 4:
        return f"{year}-12-31"
    return None


def quarter_token_from_folder(folder_name: str) -> Optional[str]:
    match = re.search(r"raw_([0-9]q\d{2})", folder_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


# -------------------------
# Parsers
# -------------------------

def parse_allianz_excel(file_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
    header_row = detect_allianz_header_row(df_raw)
    df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
    extracted_date = extract_date_from_excel(df_raw)

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
    if extracted_date:
        df["data"] = df.get("data", "")
        df.loc[df["data"].astype(str).str.strip() == "", "data"] = extracted_date
    df = df.dropna(axis=0, how="all")

    return ensure_output_schema(df)


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

    mapping = {
        "nazwa subfunduszu": "fundusz",
        "data wyceny": "data",
        "rodzaj instrumentu": "typ_aktywa",
        "emitent": "emitent",
        "kod isin": "isin",
        "waluta notowań": "waluta",
        "waluta nominału": "waluta",
        "ilość": "liczba_sztuk",
        "wartość całkowita w walucie wyceny funduszu": "wartosc_pln",
    }

    for src, dst in mapping.items():
        col = find_column(df, src)
        if col:
            df[dst] = df[col]
        else:
            df[dst] = ""

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
    try:
        df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
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
        df = pd.read_excel(file_path, engine="openpyxl", header=header_row)
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

    df = df[df["fundusz"].apply(is_not_empty) | df["isin"].apply(is_not_empty)]
    
    # Filter to keep only PPK funds
    fundusz_col = df.get("fundusz", "").astype(str)
    df = df[fundusz_col.str.contains("PPK", case=False, na=False)]
    
    df = df.dropna(axis=0, how="all")

    return ensure_output_schema(df)


def parse_esaliens_pdf(file_path: str) -> pd.DataFrame:
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(file_path))
    file_date = date_match.group(1) if date_match else ""

    def safe_page_text_simple(page) -> str:
        extract_simple = getattr(page, "extract_text_simple", None)
        if not callable(extract_simple):
            return ""

        def _timeout_handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.alarm(2)
            return extract_simple() or ""
        except TimeoutError:
            return ""
        except Exception:
            return ""
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    subfundusz_pattern = re.compile(r"\bESA\s+\d{4}\b")
    number_pattern = re.compile(r"\d{1,3}(?:\s\d{3})*,\d{2}")
    currency_pattern = re.compile(r"\b[A-Z]{3}\b")

    rows: List[Dict[str, str]] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = safe_page_text_simple(page)
            for line in page_text.splitlines():
                if not isin_pattern.search(line):
                    continue
                tokens = line.split()
                idx_isin = None
                for i, token in enumerate(tokens):
                    if isin_pattern.fullmatch(token):
                        idx_isin = i
                        break
                if idx_isin is None:
                    continue

                isin = tokens[idx_isin]
                if not any(ch.isdigit() for ch in isin):
                    continue
                typ_aktywa = tokens[idx_isin + 1] if len(tokens) > idx_isin + 1 else ""
                kraj = tokens[idx_isin + 2] if len(tokens) > idx_isin + 2 else ""
                waluta = tokens[idx_isin + 3] if len(tokens) > idx_isin + 3 else ""

                numbers: List[str] = []
                for match in number_pattern.finditer(line):
                    tail = line[match.end():].lstrip()
                    if tail.startswith("%"):
                        continue
                    numbers.append(match.group(0))
                liczba_sztuk = numbers[-2] if len(numbers) >= 2 else ""
                wartosc_pln = numbers[-1] if len(numbers) >= 1 else ""

                prefix = line[:line.find(isin)]
                subfundusz_match = subfundusz_pattern.search(prefix)
                fundusz = subfundusz_match.group(0) if subfundusz_match else ""
                emitent = ""
                if subfundusz_match:
                    prefix_after_subfundusz = prefix.split(fundusz, 1)[-1]
                    emitent = re.sub(r"\bSFIO\b", "", prefix_after_subfundusz, flags=re.IGNORECASE)
                    emitent = re.sub(r"\bPLN\b", "", emitent, flags=re.IGNORECASE)
                    emitent = re.sub(
                        r"Specjalistyczny Fundusz Inwestycyjny Otwarty",
                        "",
                        emitent,
                        flags=re.IGNORECASE,
                    )
                    emitent = re.sub(r"Inwestycyjny Otwarty", "", emitent, flags=re.IGNORECASE)
                    emitent = emitent.strip()
                else:
                    prefix_currencies = currency_pattern.findall(prefix)
                    if prefix_currencies:
                        last_curr = prefix_currencies[-1]
                        emitent = prefix.split(last_curr, 1)[-1].strip()

                if emitent and typ_aktywa and waluta and liczba_sztuk and wartosc_pln:
                    rows.append(
                        {
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
                    )

    if rows:
        df_text = pd.DataFrame(rows)
        df_text = ensure_output_schema(df_text)
        df_text = df_text.drop_duplicates(subset=OUTPUT_COLUMNS)
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

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        sample_text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
        date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", sample_text)
        if date_match:
            data = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
        else:
            data = "2025-12-30"

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

                if isin_idx is None or isin_idx < 2:
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

                value_start_idx = comma_idx
                if comma_idx >= 1 and remaining[comma_idx - 1].isdigit():
                    value_start_idx = comma_idx - 1

                wartosc_pln = " ".join(remaining[value_start_idx : comma_idx + 1])

                liczba_sztuk = " ".join(
                    [p for p in remaining[:value_start_idx] if p.isdigit()]
                )

                if not liczba_sztuk:
                    value_start_idx = comma_idx
                    wartosc_pln = remaining[comma_idx]
                    liczba_sztuk = " ".join(
                        [p for p in remaining[:value_start_idx] if p.isdigit()]
                    )

                waluta = None
                waluta_idx = -1
                for j in range(value_start_idx - 1, -1, -1):
                    if not remaining[j].isdigit():
                        waluta = remaining[j]
                        waluta_idx = j
                        break

                if waluta is None or waluta_idx < 0:
                    continue

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
                        current_fundusz = fundusz_candidate

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
    return ensure_output_schema(df)


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


# -------------------------
# Detection & main flow
# -------------------------

def detect_parser(file_path: str) -> Optional[str]:
    name = os.path.basename(file_path).lower()
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

    if "uniqa" in name and name.endswith(".pdf"):
        return "uniqa"
    
    # Generali: Excel format
    if "generali" in name and name.endswith((".xlsx", ".xls")):
        return "generali"

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

    parser_map: Dict[str, Callable[[str], pd.DataFrame]] = {
        "allianz": parse_allianz_excel,
        "santander": parse_santander_excel,
        "bnp": parse_bnp_excel,
        "goldman": parse_goldman_excel,
        "millennium": parse_millennium_excel,
        "pfr": parse_pfr_excel,
        "pko": lambda p: parse_pko_excel(p, quarter_end),
        "pzu": parse_pzu_excel,
        "esaliens": parse_esaliens_pdf,
        "generali": parse_generali_excel,
        "uniqa": parse_uniqa_pdf,
        "nn": parse_nn_pdf,
        "investors": parse_investors_pdf,
        "pekao": parse_pekao_pdf,
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
        parser_key = detect_parser(file_path)
        if not parser_key:
            continue
        parser = parser_map.get(parser_key)
        if not parser:
            continue
        if file_path.lower().endswith(".pdf"):
            if parser_key in ("generali", "uniqa", "investors", "pekao", "esaliens", "nn"):
                df = parser(file_path)
            else:
                df = run_parser_with_timeout(parser, file_path, timeout_seconds=20)
        else:
            df = parser(file_path)
        all_rows.append(df)

    if not all_rows:
        return ensure_output_schema(pd.DataFrame())

    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output_csv")
    os.makedirs(output_dir, exist_ok=True)

    raw_folders = [
        path
        for path in glob.glob(os.path.join(base_dir, "raw_*"))
        if os.path.isdir(path)
    ]

    for folder in raw_folders:
        quarter_token = quarter_token_from_folder(os.path.basename(folder)) or ""
        if not quarter_token:
            continue
        result_df = process_folder(folder)
        output_path = os.path.join(output_dir, f"PPK_{quarter_token}.csv")
        result_df = result_df.copy()
        for col in ["liczba_sztuk", "wartosc_pln"]:
            if col in result_df.columns:
                result_df[col] = result_df[col].apply(format_decimal_comma)
        result_df.to_csv(
            output_path,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
