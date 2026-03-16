from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber

PDF_TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
}

SECTION_START = "TABELE UZUPEŁNIAJĄCE"
SECTION_END = "Tabele Dodatkowe"


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_fund_year_from_name(file_name: str) -> str:
    match = re.search(r"(20\d{2})(?=\.pdf$)", file_name, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def detect_section_pages(pdf: pdfplumber.PDF) -> tuple[int | None, int | None]:
    start_page = None
    end_page = None

    for i, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text() or ""
        if start_page is None and SECTION_START in page_text:
            start_page = i
            continue
        if start_page is not None and SECTION_END in page_text:
            end_page = i
            break

    if start_page is not None and end_page is None:
        end_page = len(pdf.pages) + 1

    return start_page, end_page


def is_boilerplate_row(row_cells: Iterable[str]) -> bool:
    text = " ".join(c for c in row_cells if c).strip()
    if not text:
        return True
    upper = text.upper()
    compact = re.sub(r"[^A-Z0-9]", "", upper)

    if "ROCZNESPRAWOZDANIEJEDNOSTKOWE" in compact:
        return True
    if "TABELEUZUP" in compact:
        return True
    if "ZOSTALYZAPREZEN" in compact or "OSTALYZAPREZEN" in compact:
        return True
    if "STRONA" in compact and "INPZUPL" in compact:
        return True

    return False


def extract_tables_from_pdf(pdf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    long_rows: list[dict] = []
    wide_rows: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        start_page, end_page = detect_section_pages(pdf)
        if start_page is None or end_page is None:
            return pd.DataFrame(), pd.DataFrame(), {
                "source_file": pdf_path.name,
                "fund_year": extract_fund_year_from_name(pdf_path.name),
                "section_start_page": None,
                "section_end_page_exclusive": None,
                "pages_processed": 0,
                "tables_extracted": 0,
                "rows_extracted": 0,
                "status": "section_not_found",
            }

        table_counter = 0
        row_counter = 0

        for page_no in range(start_page, end_page):
            page = pdf.pages[page_no - 1]
            tables = page.extract_tables(table_settings=PDF_TABLE_SETTINGS) or []

            for t_idx, table in enumerate(tables, start=1):
                if not table:
                    continue

                table_counter += 1
                for r_idx, row in enumerate(table, start=1):
                    cleaned = [clean_cell(cell) for cell in (row or [])]
                    if is_boilerplate_row(cleaned):
                        continue

                    row_counter += 1
                    base = {
                        "source_file": pdf_path.name,
                        "fund_year": extract_fund_year_from_name(pdf_path.name),
                        "page": page_no,
                        "table_no_on_page": t_idx,
                        "row_no_in_table": r_idx,
                    }

                    row_wide = dict(base)
                    for c_idx, value in enumerate(cleaned, start=1):
                        row_wide[f"c{c_idx:02d}"] = value
                        long_rows.append(
                            {
                                **base,
                                "col_no": c_idx,
                                "cell_text": value,
                            }
                        )

                    wide_rows.append(row_wide)

        metadata = {
            "source_file": pdf_path.name,
            "fund_year": extract_fund_year_from_name(pdf_path.name),
            "section_start_page": start_page,
            "section_end_page_exclusive": end_page,
            "pages_processed": max(0, end_page - start_page),
            "tables_extracted": table_counter,
            "rows_extracted": row_counter,
            "status": "ok",
        }

    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows), metadata


def build_workbook(input_dir: Path, output_path: Path) -> None:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Brak plikow PDF w katalogu: {input_dir}")

    all_long: list[pd.DataFrame] = []
    all_wide: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for pdf_path in pdf_files:
        long_df, wide_df, meta = extract_tables_from_pdf(pdf_path)
        if not long_df.empty:
            all_long.append(long_df)
        if not wide_df.empty:
            all_wide.append(wide_df)
        meta_rows.append(meta)

    long_out = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame(
        columns=["source_file", "fund_year", "page", "table_no_on_page", "row_no_in_table", "col_no", "cell_text"]
    )
    wide_out = pd.concat(all_wide, ignore_index=True) if all_wide else pd.DataFrame(
        columns=["source_file", "fund_year", "page", "table_no_on_page", "row_no_in_table", "c01"]
    )
    meta_out = pd.DataFrame(meta_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        meta_out.to_excel(writer, sheet_name="meta", index=False)
        wide_out.to_excel(writer, sheet_name="supp_rows_wide", index=False)
        long_out.to_excel(writer, sheet_name="supp_cells_long", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ekstrakcja CALYCH tabel uzupelniajacych z PDF PZU 4Q24 do Excela (bez main.py)."
    )
    parser.add_argument(
        "--input-dir",
        default="raw_4Q24/4Q24_PZU1",
        help="Katalog z PDF-ami PZU1 dla 4Q24",
    )
    parser.add_argument(
        "--output",
        default="raw_4Q24/PZU1_2024-12-31_tabele_uzupelniajace.xlsx",
        help="Sciezka do wynikowego pliku Excel",
    )
    args = parser.parse_args()

    build_workbook(Path(args.input_dir), Path(args.output))
    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
