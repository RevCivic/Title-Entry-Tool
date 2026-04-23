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
| `COMPOSE_PROFILES`        | *(empty)*                    | Set to `ai` to start Ollama AI services with the stack    |
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
cp .env.example .env   # edit credentials and profile as needed
```

#### Controlling which services start

AI services are gated behind the `ai` Compose profile.  Which services start
is determined by the `COMPOSE_PROFILES` environment variable — **no CLI flags
required**.

| `COMPOSE_PROFILES` value | Services started                                      |
|--------------------------|-------------------------------------------------------|
| *(empty or unset)*       | PostgreSQL + Flask app (OCR only, starts immediately) |
| `ai`                     | Above **plus** Ollama server and model downloader     |

Set the variable in your `.env` file, in Portainer's *Environment Variables*
screen, or in any other orchestration layer that injects environment variables
before `docker compose up` runs.

```dotenv
# .env – OCR only (default)
COMPOSE_PROFILES=

# .env – with AI assistance
COMPOSE_PROFILES=ai
```

Then start the stack the same way regardless of mode:

```bash
docker compose up --build
```

> **Disk space:** allocate at least 5 GB for the `ollama_data` volume
> (`moondream` is ~1.5 GB; larger models such as `llava` require ~4 GB).

> **First AI run:** the `ollama-init` service downloads the configured model
> (~1.5–4 GB) before AI extraction becomes active.  The app continues accepting
> uploads with OCR while the download completes.

> **Image version:** set `OLLAMA_VERSION` in your `.env` (or stack environment)
> to pin a specific release tag, e.g. `OLLAMA_VERSION=0.21.1`.  Both the
> `ollama` and `ollama-init` services share this variable, so one change
> updates both.  Leave it unset (or `latest`) to always pull the newest image.

#### CLI alternative

If you prefer to keep `COMPOSE_PROFILES` empty and activate AI from the
command line:

```bash
docker compose --profile ai up --build
```

#### Optional GPU for AI service

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, layer in
the GPU override file.  `COMPOSE_PROFILES=ai` (or `--profile ai`) is still
required to start Ollama:

```bash
# env-var approach (recommended for Portainer)
COMPOSE_PROFILES=ai docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# CLI approach
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile ai up --build
```

Data is persisted in the `postgres_data` and `ollama_data` named volumes.
Application artifacts (e.g. exported CSVs) are stored in the `app_data` named
volume mounted at `/app/data`.

### Health, status, and maintenance endpoints

| Endpoint                              | Method | Description                                                                         |
|---------------------------------------|--------|-------------------------------------------------------------------------------------|
| `GET /health`                         | GET    | Returns `{"status": "ok"}` when the Flask app is running.                          |
| `GET /api/status`                     | GET    | Returns extraction provider and AI readiness (`ready` \| `loading` \| `unavailable` \| `not_configured`). |
| `GET /api/maintenance/containers`     | GET    | Returns state/health of all compose services (requires Docker socket mount).       |
| `GET /api/maintenance/logs/<service>` | GET    | Returns recent log lines for a service (`?lines=N`, default 100, max 500).        |
| `POST /api/maintenance/ai/start`      | POST   | Starts stopped `ollama` and `ollama-init` containers (requires Docker socket mount). |

> **AI status values:**
> - `ready` – Ollama API is reachable and the configured model is available.
> - `loading` – Ollama is running but the model download is still in progress.
> - `unavailable` – Ollama did not respond (services not started).
> - `not_configured` – `EXTRACTION_PROVIDER=tesseract`; AI is intentionally disabled.

The container healthcheck polls `/health` so orchestrators (Docker Compose,
Kubernetes, etc.) can detect real application readiness.

### Run tests

```bash
python -m unittest discover -s tests -v
```
