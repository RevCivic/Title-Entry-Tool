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
```

#### First 5 minutes – core services only (immediate)

```bash
docker compose up --build
```

This starts PostgreSQL and the Flask app.  The app is ready to accept uploads
immediately using OCR (Tesseract) extraction.  Open `http://localhost:8000`.

#### Add AI-assisted extraction

When you are ready to enable the AI vision model, start the `ai` profile:

```bash
docker compose --profile ai up --build
```

This adds two services:
- **`ollama`** – runs the Ollama server (starts immediately).
- **`ollama-init`** – downloads the configured model (`AI_MODEL`, default
  `moondream`) in the background.  The app continues working with OCR while the
  download completes (~2–4 GB on first run).

The status bar at the top of the UI shows the current extraction mode and
whether AI is ready. Once the model download finishes, AI-assisted extraction
activates automatically on the next upload.

> **Disk space:** allocate at least 5 GB for the `ollama_data` volume
> (`moondream` is ~1.5 GB; larger models such as `llava` require ~4 GB).

> **Image pinning:** the compose file uses `ollama/ollama:latest`.  For
> production deployments, pin to a specific release, e.g.
> `image: ollama/ollama:0.6.8`, to prevent unexpected upgrades.

#### Optional GPU for AI service

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ai up --build
```

Data is persisted in the `postgres_data` and `ollama_data` named volumes.
Application artifacts (e.g. exported CSVs) are stored in the `app_data` named
volume mounted at `/app/data`.

### Health and status endpoints

| Endpoint      | Description                                                   |
|---------------|---------------------------------------------------------------|
| `GET /health` | Returns `{"status": "ok"}` when the Flask app is running.    |
| `GET /api/status` | Returns the configured extraction provider and current AI readiness (`ready` \| `unavailable` \| `not_configured`). |

The container healthcheck polls `/health` so orchestrators (Docker Compose,
Kubernetes, etc.) can detect real application readiness.

### Run tests

```bash
python -m unittest discover -s tests -v
```
