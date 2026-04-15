# Title-Entry-Tool
A tool to expediate title entry for the NMVITIS upload system.

## Current workflow

- Record OCR-derived title data from multiple states into PostgreSQL.
- Validate title number, VIN (including check digit), and vehicle year during insert.
- Flag invalid rows with stored validation errors for correction.
- Export only validated records to CSV for NMVITIS upload.

## Quick usage

```python
from title_entry_tool import create_connection_from_env, initialize_database, insert_title_record, export_validated_to_csv

connection = create_connection_from_env()
initialize_database(connection)

insert_title_record(connection, "NM", "ABC-1234", "1HGCM82633A004352", 2003, ocr_text="OCR text")
export_validated_to_csv(connection, "nmvitis_upload.csv")
```

### Environment variables

| Variable                  | Default                      | Description                                               |
|---------------------------|------------------------------|-----------------------------------------------------------|
| `PORT`                    | `8000`                       | Port the Flask app listens on                             |
| `DEFAULT_STATE`           | `NM`                         | Fallback state when OCR finds none                        |
| `DB_HOST`                 | `localhost`                  | PostgreSQL hostname                                       |
| `DB_PORT`                 | `5432`                       | PostgreSQL port                                           |
| `DB_NAME`                 | `titles`                     | PostgreSQL database name                                  |
| `DB_USER`                 | `postgres`                   | PostgreSQL user                                           |
| `DB_PASSWORD`             | *(empty)*                    | PostgreSQL password                                       |
| `EXTRACTION_PROVIDER`     | `hybrid`                     | `tesseract`, `ai`, or `hybrid`                            |
| `AI_ENDPOINT`             | `http://ollama:11434`        | Base URL of the self-hosted Ollama service                |
| `AI_MODEL`                | `moondream`                  | Ollama model name (e.g. `moondream`, `llava`)             |
| `AI_TIMEOUT`              | `60`                         | HTTP timeout in seconds for each AI inference call        |
| `AI_CONFIDENCE_THRESHOLD` | `0.6`                        | Fields below this confidence are flagged for review       |

Copy `.env.example` to `.env` and edit values before running.

## Web application (PDF/Image upload + extraction)

This repository includes a Flask web app (`web_app.py`) that can:

- Upload PDF or image files (`pdf`, `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp`)
- Extract text from:
  - PDFs (native text; OCR fallback for image-only pages)
  - Images (OCR via Tesseract)
- Parse title fields (`state`, `title_number`, `vin`, `vehicle_year`) using Tesseract
  regex matching **and/or** a self-hosted AI vision model
- Display per-field confidence scores and flag low-confidence results for review
- Validate and store records in PostgreSQL using the existing validation logic

### Extraction providers

The app supports three extraction modes controlled by `EXTRACTION_PROVIDER`:

| Mode         | Behaviour                                                                                    |
|--------------|----------------------------------------------------------------------------------------------|
| `hybrid`     | AI is tried first; fields below the confidence threshold fall back to Tesseract (default)   |
| `ai`         | AI only; if the AI service is unreachable the app falls back to Tesseract automatically     |
| `tesseract`  | Tesseract/regex only – no AI calls made                                                      |

### Run locally

```bash
pip install -r requirements.txt
# Export required DB env vars first, then:
python web_app.py
```

To run on a different port:

```bash
PORT=5000 python web_app.py
```

Then open `http://localhost:8000` (or your configured port).

### Run with Docker Compose (recommended)

```bash
cp .env.example .env   # edit credentials as needed
docker compose up --build
```

This starts PostgreSQL, the Flask app, and Ollama automatically. The Ollama
container auto-pulls the configured model (`AI_MODEL`, default `moondream`) on
startup, so no manual `ollama pull ...` step is required.

> On first startup, model download can take a few minutes depending on network
> speed.

#### Optional GPU for AI service

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, enable GPU
assignment for Ollama with the provided override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Data is persisted in the `postgres_data` and `ollama_data` named volumes.
Application artifacts (e.g. exported CSVs) are stored in the `app_data` named
volume mounted at `/app/data`.

### Run tests

```bash
python -m unittest discover -s tests -v
```
