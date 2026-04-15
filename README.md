# Title-Entry-Tool
A tool to expediate title entry for the NMVITIS upload system.

## Current workflow

- Record OCR-derived title data from multiple states into SQLite.
- Validate title number, VIN (including check digit), and vehicle year during insert.
- Flag invalid rows with stored validation errors for correction.
- Export only validated records to CSV for NMVITIS upload.

## Quick usage

```python
from title_entry_tool import create_connection, initialize_database, insert_title_record, export_validated_to_csv

connection = create_connection("titles.db")
initialize_database(connection)

insert_title_record(connection, "NM", "ABC-1234", "1HGCM82633A004352", 2003, ocr_text="OCR text")
export_validated_to_csv(connection, "nmvitis_upload.csv")
```

## Web application (PDF/Image upload + extraction)

This repository now includes a Flask web app (`web_app.py`) that can:

- Upload PDF or image files (`pdf`, `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp`)
- Extract text from:
  - PDFs (native text; OCR fallback for image-only pages)
  - Images (OCR via Tesseract)
- Parse title fields (`state`, `title_number`, `vin`, `vehicle_year`)
- Validate and store records in SQLite using the existing validation logic

### Run locally

```bash
pip install -r requirements.txt
python web_app.py
```

Then open `http://localhost:8000`.

### Run with Docker

```bash
docker build -t title-entry-tool .
docker run --rm -p 8000:8000 -v "$(pwd)":/app title-entry-tool
```
