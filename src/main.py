import os
import glob
import re
import signal
import multiprocessing as mp
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
import pdfplumber
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pragma: no cover
    pdfminer_extract_text = None

OUTPUT_COLUMNS = [
    "data",
    "instytucja",
    "fundusz",
    "typ_aktywa",
    "emitent",
    "isin",
    "waluta",
    "liczba_sztuk",
    "wartosc_pln",
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


def parse_polish_number(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if text == "":
        return 0.0
    text = text.replace(" ", "")
    text = text.replace(".", "") if "," in text else text
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def format_decimal_comma(value) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).replace(".", ",")

    text = f"{number:.2f}"
    return text.replace(".", ",")


def safe_string(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]
    
    # Convert data column to YYYY-MM-DD format if it's datetime
    if "data" in df.columns:
        df["data"] = df["data"].apply(lambda x: pd.to_datetime(x, errors="coerce").strftime("%Y-%m-%d") if pd.notna(x) and str(x).strip() != "" else "")
    
    df["liczba_sztuk"] = df["liczba_sztuk"].apply(parse_polish_number)
    df["wartosc_pln"] = df["wartosc_pln"].apply(parse_polish_number)
    df = df.fillna("")
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
                if token in sheet:
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

    rows: List[Dict[str, str]] = []
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
                    if "uniqa" not in joined:
                        continue

                    raw_isin = None
                    for value in row_values:
                        match = isin_pattern.search(value or "")
                        if match:
                            raw_isin = match.group(0)
                            break
                    if not raw_isin:
                        joined_raw = " ".join([v for v in row_values if v])
                        match_loose = isin_loose_pattern.search(joined_raw)
                        if match_loose:
                            raw_isin = match_loose.group(0)
                    if not raw_isin:
                        continue

                    fundusz = _clean_fundusz(row_values[2]) if len(row_values) > 2 else ""
                    waluta = _clean_nd(row_values[5]).strip() if len(row_values) > 5 else ""
                    emitent = _clean_emitent(row_values[6]) if len(row_values) > 6 else ""
                    isin = _clean_isin(row_values[7]) if len(row_values) > 7 else ""
                    typ_aktywa = _clean_typ_aktywa(row_values[9]) if len(row_values) > 9 else ""
                    liczba_sztuk = _clean_nd(row_values[12]) if len(row_values) > 12 else ""
                    wartosc_pln = _clean_nd(row_values[13]) if len(row_values) > 13 else ""

                    if not isin_pattern.fullmatch(isin):
                        isin = _clean_isin_loose(raw_isin)

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

                    if wartosc_num == 0:
                        continue

                    rows.append(
                        {
                            "data": file_date,
                            "instytucja": "UNIQA TFI S.A.",
                            "fundusz": fundusz,
                            "typ_aktywa": typ_aktywa,
                            "emitent": emitent,
                            "isin": isin,
                            "waluta": waluta,
                            "liczba_sztuk": parse_polish_number(liczba_sztuk),
                            "wartosc_pln": wartosc_num,
                        }
                    )

    df = pd.DataFrame(rows)
    return ensure_output_schema(df)


def parse_generali_pdf(file_path: str, fallback_date: str) -> pd.DataFrame:
    preview_text = extract_text_pdfminer(file_path, page_numbers=[0], timeout_seconds=10)
    file_date = parse_date_from_text(preview_text) or fallback_date

    rows: List[Dict[str, str]] = []
    isin_pattern = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    isin_spaced_pattern = re.compile(r"(?:[A-Z]\s*){2}(?:[A-Z0-9]\s*){9}(?:\s*\d)")
    fx_code_pattern = re.compile(
        r"\b(?:DFX|FX)[A-Z]{6}\d{3}\d{5}N\d{3}\b|\bDFX[A-Z]{6}\d{3}\b"
    )
    e00_pattern = re.compile(r"\bE00[A-Z0-9]{8}\b")
    isin_ie_pattern = re.compile(r"I\s*E\s*0\s*0\s*(?:[A-Z0-9]\s*){8}")
    number_pattern = re.compile(r"\d{1,3}(?:\s\d{3})*,\d{2}")
    currency_pattern = re.compile(r"\b[A-Z]{3}\b")
    fundusz_pattern = re.compile(r"(Generali\s+Horyzont\s+\d{4})")
    typ_keywords = [
        "Jednostki uczestnictwa",
        "List zastawny",
        "Papier komercyjny",
        "Obligacja",
        "Akcja",
        "FX",
    ]

    def parse_text_lines(page_text: str) -> None:
        for line in page_text.splitlines():
                line = line.strip()
                if not line or not re.match(r"^\d+", line):
                    continue
                fx_match = fx_code_pattern.search(line)
                ie_match = isin_ie_pattern.search(line)
                isin_match = isin_pattern.search(line)
                if not isin_match:
                    isin_match = isin_spaced_pattern.search(line)
                e00_match = None
                if not isin_match and not fx_match and not ie_match:
                    e00_match = e00_pattern.search(line)

                chosen_match = None
                if fx_match and (not isin_match or fx_match.start() < isin_match.start()):
                    chosen_match = fx_match
                elif ie_match:
                    chosen_match = ie_match
                else:
                    chosen_match = isin_match or e00_match

                if not chosen_match:
                    continue

                raw_isin = chosen_match.group(0)
                isin = re.sub(r"\s+", "", raw_isin)
                if e00_match and chosen_match == e00_match:
                    isin = f"I{isin}"
                if ie_match and chosen_match == ie_match:
                    isin = isin.replace("IE00", "IE00")
                if isin_match and "UCITS" in line.upper() and not isin.startswith("IE"):
                    if ie_match:
                        isin = re.sub(r"\s+", "", ie_match.group(0))

                fundusz_match = fundusz_pattern.search(line)
                fundusz = fundusz_match.group(1) if fundusz_match else ""

                prefix = line[:chosen_match.start()]
                prefix_currencies = currency_pattern.findall(prefix)
                emitent = ""
                if "FX" in raw_isin or re.search(r"\bFX\b", line):
                    pair_match = re.search(r"\b[A-Z]{3}/[A-Z]{3}\b", line)
                    emitent = pair_match.group(0) if pair_match else ""
                else:
                    if prefix_currencies:
                        last_curr = prefix_currencies[-1]
                        emitent = prefix.split(last_curr)[-1].strip()
                    else:
                        emitent = prefix.strip()
                    emitent = re.sub(
                        r"Specjalistyczny Fundusz Inwestycyjny Otwarty",
                        "",
                        emitent,
                        flags=re.IGNORECASE,
                    )
                    emitent = re.sub(r"\bN/D\b", "", emitent, flags=re.IGNORECASE).strip()
                    emitent = re.sub(r"\b[A-Z]{3}\b", "", emitent).strip()
                    emitent = re.sub(r"^\s*\d+[A-Z0-9]*\s+", "", emitent).strip()
                    if fundusz and fundusz in emitent:
                        emitent = emitent.replace(fundusz, "").strip()
                    emitent = re.sub(r"\bUCITS?\b", "", emitent, flags=re.IGNORECASE).strip()

                after = line[chosen_match.end():].strip()
                after_tokens = after.split()
                typ_aktywa = ""
                for keyword in typ_keywords:
                    if re.search(rf"\b{re.escape(keyword)}\b", line):
                        typ_aktywa = keyword
                        break
                if not typ_aktywa and ("FX" in raw_isin or re.search(r"\bFX\b", line)):
                    typ_aktywa = "FX"
                if not typ_aktywa and after_tokens:
                    if after_tokens[0] == "N/D" and len(after_tokens) > 1:
                        typ_aktywa = after_tokens[1]
                        if len(after_tokens) > 2 and after_tokens[2].islower():
                            typ_aktywa = f"{typ_aktywa} {after_tokens[2]}"
                    else:
                        typ_aktywa = after_tokens[0]

                all_currencies = currency_pattern.findall(line)
                waluta = all_currencies[-1] if all_currencies else ""

                numbers: List[str] = []
                for match in number_pattern.finditer(line):
                    tail = line[match.end():].lstrip()
                    if tail.startswith("%"):
                        continue
                    numbers.append(match.group(0))
                liczba_sztuk = numbers[-2] if len(numbers) >= 2 else ""
                wartosc_pln = numbers[-1] if len(numbers) >= 1 else ""

                rows.append(
                    {
                        "data": file_date,
                        "instytucja": "Generali Investments TFI S.A.",
                        "fundusz": fundusz,
                        "typ_aktywa": typ_aktywa,
                        "emitent": emitent,
                        "isin": isin,
                        "waluta": waluta,
                        "liczba_sztuk": liczba_sztuk,
                        "wartosc_pln": wartosc_pln,
                    }
                )

    full_text = extract_text_pdfminer(file_path, timeout_seconds=120)
    if full_text:
        for chunk in (full_text.split("\f") if "\f" in full_text else [full_text]):
            parse_text_lines(chunk)

    if not rows:
        def safe_page_text_simple(page) -> str:
            extract_simple = getattr(page, "extract_text_simple", None)
            if not callable(extract_simple):
                return ""

            def _timeout_handler(signum, frame):
                raise TimeoutError()

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            try:
                signal.alarm(5)
                return extract_simple() or ""
            except TimeoutError:
                return ""
            except Exception:
                return ""
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = safe_page_text_simple(page)
                    if page_text:
                        parse_text_lines(page_text)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return ensure_output_schema(df)

    emitent_series = df["emitent"].astype(str) if "emitent" in df else pd.Series([""] * len(df))
    isin_series = df["isin"].astype(str) if "isin" in df else pd.Series([""] * len(df))
    typ_series = df["typ_aktywa"].astype(str) if "typ_aktywa" in df else pd.Series([""] * len(df))

    df = df[
        (emitent_series.str.strip() != "")
        | (isin_series.str.strip() != "")
        | (typ_series.str.strip() != "")
    ]
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
    if "goldman" in name and name.endswith((".xls", ".xlsx")):
        return "goldman"
    if "millennium" in name and name.endswith((".xls", ".xlsx")):
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
            if any("kod isin" in h for h in headers_normalized) and any("wartość" in h and "wyceny" in h for h in headers_normalized):
                return "pfr"
        except Exception:
            pass

    if "uniqa" in name and name.endswith(".pdf"):
        return "uniqa"

    if name.endswith(".pdf"):
        if "esaliens" in name:
            return "esaliens"
        if "generali" in name or ("horyzont" in name and "sfio" in name):
            return "generali"

        try:
            text = extract_text_pdfminer(file_path, page_numbers=[0], timeout_seconds=8).lower()
        except Exception:
            text = ""
        if "nazwa funduszu / subfunduszu" in text or "generali" in text:
            return "generali"
        if "nazwa funduszu" in text or "esaliens" in text:
            return "esaliens"

    return None


def process_folder(folder_path: str) -> pd.DataFrame:
    quarter_end = quarter_end_date_from_folder(os.path.basename(folder_path)) or ""

    parser_map: Dict[str, Callable[[str], pd.DataFrame]] = {
        "allianz": parse_allianz_excel,
        "santander": parse_santander_excel,
        "goldman": parse_goldman_excel,
        "millennium": parse_millennium_excel,
        "pfr": parse_pfr_excel,
        "pko": lambda p: parse_pko_excel(p, quarter_end),
        "pzu": parse_pzu_excel,
        "esaliens": parse_esaliens_pdf,
        "generali": lambda p: parse_generali_pdf(p, quarter_end),
        "uniqa": parse_uniqa_pdf,
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
            if parser_key in ("generali", "uniqa"):
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
