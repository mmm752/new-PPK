import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


def parse_quarter_token(token: str) -> Optional[Tuple[int, int]]:
    if not token:
        return None
    match = re.match(r"^([1-4])Q(\d{2})$", token.strip().upper())
    if not match:
        return None
    quarter = int(match.group(1))
    year = 2000 + int(match.group(2))
    return quarter, year


def quarter_sort_key(token: str) -> tuple:
    parsed = parse_quarter_token(token)
    if not parsed:
        return (9999, 99)
    quarter, year = parsed
    return (year, quarter)


def prev_quarter_token(token: str) -> Optional[str]:
    parsed = parse_quarter_token(token)
    if not parsed:
        return None
    quarter, year = parsed
    return f"{quarter}Q{(year - 1) % 100:02d}"


def quarter_token_from_folder(folder_name: str) -> Optional[str]:
    if not folder_name:
        return None
    match = re.search(r"raw_([0-9]q\d{2})", folder_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def quarter_end_date_from_folder(folder_name: str) -> Optional[str]:
    if not folder_name:
        return None
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


def quarter_token_from_date(date_like) -> Optional[str]:
    if date_like is None:
        return None
    try:
        dt = pd.to_datetime(date_like, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    year = int(dt.year)
    month = int(dt.month)
    quarter = ((month - 1) // 3) + 1
    return f"{quarter}Q{str(year % 100).zfill(2)}"
