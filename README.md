# Title-Entry-Tool
A tool to expedite title entry for the NMVITIS upload system.

## Current workflow

- Upload PDF/image files; barcode scanning, image preprocessing (deskew, Otsu binarization, perspective correction), and AI extraction pull **all NMVITIS-required fields** automatically.
- State-specific layout templates (`state_templates/`) guide targeted crops for NM, TX, CA, AZ, CO (extend by adding a JSON file).
- Two-pass AI extraction: pass 1 anchors the state and title number; pass 2 uses a state-aware prompt plus template-guided crops.
- Validate title number, VIN (including check digit), and vehicle year during insert.
- Export validated records to a full NMVITIS-aligned CSV (all required columns).
- Review and correct extracted fields in a web UI (`/review`); approve records as ground truth.
- Generate per-record NMVITIS PDF submission forms (`/api/generate-pdf/<id>`).
- Import NMVITIS rejection CSVs back as flagged corrections (`/import-rejections`).
- Export ground-truth corrections as training data (`python train_ocr.py export`).

## Quick usage

```python
from title_entry_tool import (
    create_connection_from_env, initialize_database,
    insert_title_record, export_validated_to_csv
)

connection = create_connection_from_env()
initialize_database(connection)

insert_title_record(
    connection, state="NM", title_number="ABC-1234",
    vin="1HGCM82633A004352", vehicle_year=2003,
    make="HONDA", model="ACCORD", color="SILVER", odometer=45000,
    owner_name="JANE DOE", owner_address="123 MAIN ST",
)
export_validated_to_csv(connection, "nmvitis_upload.csv")
```

### Environment variables

| Variable                  | Default                      | Description                                                   |
|---------------------------|------------------------------|---------------------------------------------------------------|
| `COMPOSE_PROFILES`        | *(empty)*                    | Set to `ai` to start Ollama AI services with the stack        |
| `PORT`                    | `8000`                       | Port the Flask app listens on                                 |
| `DEFAULT_STATE`           | `NM`                         | Fallback state when OCR finds none                            |
| `DATA_DIR`                | `/app/data`                  | Root directory for uploads, PDFs, and training data volumes   |
| `DB_HOST`                 | `localhost`                  | PostgreSQL hostname                                           |
| `DB_PORT`                 | `5432`                       | PostgreSQL port                                               |
| `DB_NAME`                 | `titles`                     | PostgreSQL database name                                      |
| `DB_USER`                 | `postgres`                   | PostgreSQL user                                               |
| `DB_PASSWORD`             | *(empty)*                    | PostgreSQL password                                           |
| `EXTRACTION_PROVIDER`     | `hybrid`                     | `tesseract`, `ai`, or `hybrid`                                |
| `AI_ENDPOINT`             | `http://ollama:11434`        | Base URL of the self-hosted Ollama service                    |
| `AI_MODEL`                | `moondream`                  | Ollama model name used for extraction (`moondream` recommended for documents) |
| `LLM_BASE_MODEL`          | *(empty — falls back to `AI_MODEL`)* | Base model for Training → Apply to AI Model. **Required** when `AI_MODEL` is already set to a custom model name (e.g. `titles-custom`). Set to the underlying registry model (e.g. `moondream`). |
| `AI_TIMEOUT`              | `60`                         | HTTP timeout in seconds for each AI inference call            |
| `AI_CONFIDENCE_THRESHOLD` | `0.6`                        | Fields below this confidence are flagged for review           |

Copy `.env.example` to `.env` and edit values before running.

## Web application (PDF/Image upload + extraction)

This repository includes a Flask web app (`web_app.py`) that can:

- Upload PDF or image files (`pdf`, `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp`)
- Scan barcodes (PDF417, QR, DataMatrix) for high-confidence VIN/title-number anchors
- Preprocess images (deskew, Otsu binarization, document-border detection)
- Extract text from:
  - PDFs (native text; OCR fallback for image-only pages)
  - Images (OCR via Tesseract with preprocessing)
- Extract all NMVITIS required fields using a two-pass AI strategy plus Tesseract fallback
- Display per-field confidence scores and flag low-confidence results for review
- Validate and store records in PostgreSQL
- Serve a review/correction UI, back-of-title form, and downloadable PDF

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

> **Disk space:** allocate at least 2 GB for the `ollama_data` volume
> (`moondream` is ~1.8 GB; alternative `qwen2.5vl:3b` requires ~2.5 GB).

> **First AI run:** the `ollama-init` service downloads the configured model
> before AI extraction becomes active.  The app continues accepting
> uploads with OCR while the download completes.
>
> If AI setup does not complete, review both startup and model-pull logs:
>
> ```bash
> docker compose logs ollama
> docker compose logs ollama-init
> ```

> **Image version:** set `OLLAMA_VERSION` in your `.env` (or stack environment)
> to pin a specific release tag, e.g. `OLLAMA_VERSION=0.21.1`.

#### CLI alternative

If you prefer to keep `COMPOSE_PROFILES` empty and activate AI from the
command line:

```bash
docker compose --profile ai up --build
```

#### Optional GPU for AI service

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, layer in
the GPU override file:

```bash
COMPOSE_PROFILES=ai docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Data is persisted in the `postgres_data` and `ollama_data` named volumes.
Application artifacts (uploads, exported CSVs, generated PDFs) are stored in
the `app_data` named volume mounted at `/app/data`.

### All endpoints

| Endpoint                              | Method       | Description                                                                              |
|---------------------------------------|--------------|------------------------------------------------------------------------------------------|
| `GET /`                               | GET          | Upload form                                                                               |
| `POST /`                              | POST         | Upload title image/PDF; extract all fields; save record                                   |
| `GET /review`                         | GET          | List all title records for review/correction; date-filtered CSV export control            |
| `GET /review/<id>`                    | GET          | Show a single record side-by-side with its source image                                   |
| `POST /review/<id>`                   | POST         | Submit field corrections; optional ground-truth approval                                   |
| `GET /titles/<id>/back`               | GET          | Show the back-of-title form for a record                                                   |
| `POST /titles/<id>/back`              | POST         | Save back-of-title data (odometer, buyer, seller)                                          |
| `GET /export`                         | GET          | Download NMVITIS-aligned CSV (`?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD`)                |
| `GET /import-rejections`              | GET          | NMVITIS rejection import form                                                              |
| `POST /import-rejections`             | POST         | Upload a NMVITIS rejection CSV; flags affected records for correction                      |
| `GET /api/generate-pdf/<id>`          | GET          | Generate and download the NMVITIS submission PDF for a record                              |
| `GET /uploads/<filename>`             | GET          | Serve a saved upload for display in the review UI                                          |
| `GET /api/training-data`              | GET          | Return ground-truth corrections as JSON for training pipeline use                          |
| `GET /health`                         | GET          | Returns `{"status": "ok"}` when the Flask app is running                                  |
| `GET /api/status`                     | GET          | Returns extraction provider and AI readiness                                               |
| `GET /api/maintenance/containers`     | GET          | Returns state/health of all compose services (requires Docker socket mount)                |
| `GET /api/maintenance/logs/<service>` | GET          | Returns recent log lines for a service                                                     |
| `POST /api/maintenance/ai/start`      | POST         | Starts stopped `ollama` and `ollama-init` containers                                       |

### State layout templates

Templates in `state_templates/` define per-field bounding-box regions as
fractions of document width/height.  Included states: **NM, TX, CA, AZ, CO**.

Add a new state by creating `state_templates/<STATE>.json`:

```json
{
  "state": "FL",
  "version": "1.0",
  "fields": {
    "vin": {
      "label_hints": ["VEHICLE IDENTIFICATION NUMBER"],
      "regions": [
        { "left": 0.0, "top": 0.08, "right": 1.0, "bottom": 0.20 }
      ]
    }
  }
}
```

### Training data export (Phase 2)

After operators approve corrections in the `/review` UI, export ground-truth
data for AI fine-tuning:

```bash
# Show statistics
python train_ocr.py stats

# Export all approved corrections as image/label pairs
python train_ocr.py export --output training_data/

# Export only specific fields
python train_ocr.py export --fields vin,make --output training_data/
```

The exported `manifest.json` is compatible with LLaVA-style LoRA fine-tuning
and Tesseract `.box`/`.tif` training pipelines.

### Custom AI model creation (prompt-tuning)

The Training dashboard lets you build a custom Ollama model that is
pre-loaded with your site's ground-truth corrections as few-shot examples.
This is **prompt-tuning only** — no GPU or weight modifications required.

#### How it works

The app calls Ollama's `/api/create` REST endpoint directly with a JSON
payload containing:
- `from` — the base registry model (e.g. `moondream`)
- `system` — a detailed NMVITIS-field extraction prompt
- `messages` — up to 30 user/assistant few-shot pairs from your approved corrections

**No Modelfile is written to disk.** Ollama stores the resulting custom model
inside the `ollama_data` Docker volume at `/root/.ollama/models/` on the
Ollama container.  You cannot reference a file path with `ollama create -f`
for this workflow; the creation happens entirely through the API.

#### End-to-end workflow

1. **Start with the base model** — ensure `AI_MODEL=moondream` (or another
   public Ollama model) in your `.env` and start the stack with `COMPOSE_PROFILES=ai`.
   The `ollama-init` service pulls the model automatically.

2. **Annotate titles** — upload PDFs/images and correct any mis-extracted
   fields in the `/review` UI.  Click **Approve** to mark records as
   ground-truth.

3. **Apply to AI Model** — go to the Training dashboard (`/training`) and
   click **Apply to AI Model**.  The app creates a new custom model (default
   name: `titles-custom`) from `moondream` plus your annotations.

4. **Switch to the custom model** — update your `.env`:

   ```dotenv
   AI_MODEL=titles-custom
   LLM_BASE_MODEL=moondream
   ```

   Then restart the stack (`docker compose up --build`).  The `ollama-init`
   service will detect that `titles-custom` is a custom (non-registry) model,
   skip the registry pull, and ensure `moondream` is available so that future
   runs of **Apply to AI Model** can rebuild the custom model at any time.

5. **Re-apply after new annotations** — repeat step 2–3 as you collect more
   corrections.  Each **Apply to AI Model** call rebuilds and overwrites the
   custom model in place.

#### Required environment variables for custom-model workflows

| Scenario | `AI_MODEL` | `LLM_BASE_MODEL` |
|---|---|---|
| Using only the public base model | `moondream` | *(leave empty)* |
| Custom model active, base still needed for re-training | `titles-custom` | `moondream` |

> **Important:** if `AI_MODEL` and `LLM_BASE_MODEL` are both set to the same
> value the app will reject the **Apply to AI Model** request with a
> configuration error.  Always point `LLM_BASE_MODEL` at the underlying
> registry model, not the custom model name.

#### Where Ollama stores models

Ollama stores all model blobs inside the `ollama_data` named Docker volume,
which is mounted into the Ollama container at `/root/.ollama/`.  The layout is:

```
/root/.ollama/
  models/
    blobs/       ← raw layer blobs (shared between model variants)
    manifests/   ← per-model manifest files (registry + custom)
      registry.ollama.ai/library/moondream/latest
      …
```

Custom models created via the API are also stored here.  The volume persists
across container restarts and stack upgrades.  To list all models currently in
the volume run:

```bash
docker compose exec ollama ollama list
```

#### Troubleshooting "file does not exist" / manifest errors

These errors mean Ollama cannot locate the **base model** when trying to
create the custom model.  Common causes and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `file does not exist` or `manifest not found` when clicking **Apply to AI Model** | `moondream` (or the model named in `LLM_BASE_MODEL`) is not present in the `ollama_data` volume | Pull the base model: `docker compose exec ollama ollama pull moondream` |
| `ollama-init` logs show the model pull failed | `AI_MODEL` is set to a custom model name before running **Apply to AI Model** for the first time | Set `LLM_BASE_MODEL=moondream` in `.env` and restart the stack; `ollama-init` will then pull `moondream` automatically |
| `AI_MODEL is set to 'titles-custom', which cannot be used as its own base model` | `LLM_BASE_MODEL` is empty or matches `AI_MODEL` | Set `LLM_BASE_MODEL=moondream` in `.env` |
| Custom model does not survive a stack restart | `ollama_data` volume was deleted or re-created | Re-run **Apply to AI Model** from the Training dashboard to rebuild it |

To manually pull the base model without restarting the stack:

```bash
docker compose exec ollama ollama pull moondream
```

To verify which models are currently available:

```bash
docker compose exec ollama ollama list
```

### Run tests

```bash
python -m unittest discover -s tests -v
```


