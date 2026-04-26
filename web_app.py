import hashlib
import io
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import pytesseract
import requests
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from PIL import Image, ImageFilter, ImageOps

from ai_extraction import (
    _ALL_FIELDS as AI_ALL_FIELDS,
    _CORE_FIELDS as AI_CORE_FIELDS,
    extract_fields_with_ai,
)
from barcode_scanner import scan_barcodes
from image_preprocessing import binarize_image, preprocess_title_image
from title_entry_tool import (
    create_connection_from_env,
    get_annotation_queue,
    get_corrections_for_record,
    get_record_by_id,
    get_title_back_record,
    get_training_stats,
    import_nmvitis_rejections,
    initialize_database,
    insert_correction,
    insert_title_back_record,
    insert_title_record,
    insert_training_run,
    list_ground_truth_corrections,
    list_records,
    list_training_runs,
    update_title_record_fields,
    export_validated_to_csv_by_date,
)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
DEFAULT_STATE = "NM"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MISSING_YEAR_SENTINEL = 0
DEFAULT_PORT = 8000

# Directory for persisted uploads and generated PDFs.
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")

# Extraction provider: "tesseract" | "ai" | "hybrid" (default)
EXTRACTION_PROVIDER = os.getenv("EXTRACTION_PROVIDER", "hybrid").lower().strip()
try:
    AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.6"))
except ValueError:
    AI_CONFIDENCE_THRESHOLD = 0.6

# All field names recognised by the extraction pipeline (core + extended).
_FIELD_NAMES: Tuple[str, ...] = AI_ALL_FIELDS

# AI readiness cache – avoids a network probe on every page load.
_AI_STATUS_CACHE: Dict[str, object] = {"status": "unknown", "checked_at": 0.0}
_AI_STATUS_TTL = 30.0  # seconds between live probes

# ── Docker / maintenance ──────────────────────────────────────────────────────

_DOCKER_SOCKET = "/var/run/docker.sock"
# Project name injected by compose; used to filter containers by label.
_COMPOSE_PROJECT = os.getenv("COMPOSE_PROJECT_NAME", "title-entry-tool")
# The four services this app is aware of.
_MANAGED_SERVICES = ("title-entry-tool", "postgres", "ollama", "ollama-init")


def _docker_client():
    """Return a Docker SDK client, or None when the socket is unavailable."""
    if not os.path.exists(_DOCKER_SOCKET):
        return None
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.DockerClient(base_url=f"unix://{_DOCKER_SOCKET}")
        client.ping()
        return client
    except Exception:
        return None


def _get_service_containers(client) -> Dict[str, Any]:
    """Return a dict keyed by service name with container state info."""
    services: Dict[str, Any] = {name: None for name in _MANAGED_SERVICES}
    try:
        containers = client.containers.list(
            all=True,
            filters={"label": f"com.docker.compose.project={_COMPOSE_PROJECT}"},
        )
        for container in containers:
            svc = container.labels.get("com.docker.compose.service")
            if svc not in services:
                continue
            health_status = ""
            health = (container.attrs.get("State") or {}).get("Health") or {}
            if health:
                health_status = health.get("Status", "")
            services[svc] = {
                "id": container.short_id,
                "name": container.name,
                "state": container.status,  # running | exited | created | …
                "health": health_status,    # healthy | unhealthy | starting | ""
            }
    except Exception:
        pass
    return services


def _get_service_logs(client, service: str, lines: int = 100) -> Dict[str, Any]:
    """Return the most-recent *lines* log lines for a compose service."""
    try:
        containers = client.containers.list(
            all=True,
            filters={
                "label": [
                    f"com.docker.compose.project={_COMPOSE_PROJECT}",
                    f"com.docker.compose.service={service}",
                ]
            },
        )
        if not containers:
            return {
                "lines": [],
                "error": (
                    f"No container found for service '{service}'. "
                    "The container may have been removed after a failed deployment."
                ),
            }

        # Prefer the newest container first in case old service revisions remain.
        containers.sort(
            key=lambda container: (container.attrs.get("Created") or ""),
            reverse=True,
        )

        container_retrieval_errors: List[str] = []
        for container in containers:
            try:
                raw = container.logs(tail=lines, timestamps=True).decode(
                    "utf-8", errors="replace"
                )
                return {
                    "lines": [ln for ln in raw.splitlines() if ln.strip()],
                    "error": None,
                    "container": {
                        "id": container.short_id,
                        "name": container.name,
                        "state": container.status,
                    },
                }
            except Exception as exc:
                container_retrieval_errors.append(
                    f"{container.name} ({container.short_id}): {type(exc).__name__}"
                )

        return {
            "lines": [],
            "error": (
                "Failed to retrieve logs from available containers. "
                f"Details: {'; '.join(container_retrieval_errors)}"
            ),
        }
    except Exception:
        return {
            "lines": [],
            "error": "Failed to retrieve logs. Check Docker socket availability and container lifecycle events.",
        }


def _start_ai_services(client) -> Dict[str, Any]:
    """Start stopped ollama / ollama-init containers if they exist."""
    messages: List[str] = []
    errors: List[str] = []
    for service in ("ollama", "ollama-init"):
        try:
            containers = client.containers.list(
                all=True,
                filters={
                    "label": [
                        f"com.docker.compose.project={_COMPOSE_PROJECT}",
                        f"com.docker.compose.service={service}",
                    ]
                },
            )
            if not containers:
                errors.append(
                    f"Container for '{service}' not found. "
                    "Set COMPOSE_PROFILES=ai and redeploy the stack to create it."
                )
                continue
            container = containers[0]
            if container.status == "running":
                messages.append(f"'{service}' is already running.")
            else:
                container.start()
                messages.append(f"'{service}' started.")
        except Exception:
            errors.append(f"Could not start '{service}'. Check Docker socket and container status.")
    return {"success": len(errors) == 0, "messages": messages, "errors": errors}


# ── AI status ─────────────────────────────────────────────────────────────────


def _check_ai_status() -> str:
    """Probe the configured AI endpoint and return its readiness.

    Returns one of:
        "not_configured" – provider is "tesseract"; AI calls are never made.
        "ready"          – Ollama API responded and the configured model exists.
        "loading"        – Ollama API responded but the model is not yet present
                           (download in progress or not yet started).
        "unavailable"    – AI is configured but the endpoint did not respond.

    Results are cached for ``_AI_STATUS_TTL`` seconds so repeated page loads
    don't each make a separate network call.
    """
    now = time.monotonic()
    if now - float(_AI_STATUS_CACHE["checked_at"]) < _AI_STATUS_TTL:
        return str(_AI_STATUS_CACHE["status"])

    if EXTRACTION_PROVIDER == "tesseract":
        status = "not_configured"
    else:
        endpoint = os.getenv("AI_ENDPOINT", "http://ollama:11434")
        model_base = os.getenv("AI_MODEL", "moondream").rsplit(":", 1)[0]
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=2)
            response.raise_for_status()
            tags_data = response.json()
            available = [
                m.get("name", "").rsplit(":", 1)[0]
                for m in tags_data.get("models", [])
            ]
            status = "ready" if model_base in available else "loading"
        except Exception:
            status = "unavailable"

    _AI_STATUS_CACHE["status"] = status
    _AI_STATUS_CACHE["checked_at"] = now
    return status


def _parse_pull_progress_from_logs(log_lines: List[str]) -> Dict[str, Any]:
    """Parse Ollama model-pull progress from ``ollama-init`` container log lines.

    Ollama streams JSON objects during a pull, one per line.  Docker timestamps
    prepend each line so the parser skips to the first ``{`` before decoding.
    Layer totals/completed values are accumulated across all digests to produce
    an aggregate download percentage.

    Returns a dict with keys:
        percent        – 0-100 float, or None when total is unknown
        status_message – last human-readable status string seen in the logs
        current_bytes  – total bytes downloaded so far
        total_bytes    – total bytes expected across all layers
        layers         – number of distinct layer digests seen
    """
    layer_totals: Dict[str, int] = {}
    layer_completed: Dict[str, int] = {}
    last_status = ""

    for line in log_lines:
        line = line.strip()
        try:
            json_start = line.index("{")
            json_part = line[json_start:]
        except ValueError:
            # Plain-text log line – keep it as a fallback status message.
            if line:
                last_status = line
            continue
        try:
            data = json.loads(json_part)
        except Exception:
            if line:
                last_status = line
            continue

        status = data.get("status", "")
        if status:
            last_status = status

        digest = data.get("digest", "")
        total = data.get("total")
        completed = data.get("completed")
        if digest and total is not None:
            layer_totals[digest] = int(total)
            layer_completed[digest] = int(completed) if completed is not None else 0

    total_bytes = sum(layer_totals.values())
    current_bytes = sum(layer_completed.values())
    percent = round(current_bytes / total_bytes * 100, 1) if total_bytes > 0 else None

    return {
        "percent": percent,
        "status_message": last_status,
        "current_bytes": current_bytes,
        "total_bytes": total_bytes,
        "layers": len(layer_totals),
    }


def _get_ai_pull_progress() -> Dict[str, Any]:
    """Return model-pull progress by examining recent ``ollama-init`` logs."""
    client = _docker_client()
    if client is None:
        return {
            "available": False,
            "percent": None,
            "status_message": "Docker socket not available",
            "current_bytes": 0,
            "total_bytes": 0,
            "layers": 0,
        }
    logs_result = _get_service_logs(client, "ollama-init", lines=300)
    if logs_result.get("error") or not logs_result.get("lines"):
        return {
            "available": True,
            "percent": None,
            "status_message": logs_result.get("error") or "No log output available",
            "current_bytes": 0,
            "total_bytes": 0,
            "layers": 0,
        }
    progress = _parse_pull_progress_from_logs(logs_result["lines"])
    progress["available"] = True
    return progress


# ── Text / field extraction ───────────────────────────────────────────────────

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
YEAR_PATTERN = re.compile(r"\b(18[8-9]\d|19\d{2}|20\d{2}|21\d{2})\b")
TITLE_PATTERN = re.compile(
    r"\bTITLE\s*(?:NO|NUMBER|#)?\s*[:\-]?\s*([A-Z0-9\-]{3,20})\b",
    re.IGNORECASE,
)
STATE_PATTERN = re.compile(r"\bSTATE\s*[:\-]?\s*([A-Z]{2})\b", re.IGNORECASE)
# High-priority pattern for 8-digit numeric title numbers (common in many US states).
TITLE_8DIGIT_PATTERN = re.compile(r"\b(\d{8})\b")
# Translation table for common OCR misreads in numeric contexts (O→0, I→1).
_OCR_NOISE_TABLE = str.maketrans("OI", "01")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_extracted_fields(text: str) -> Dict[str, Optional[Union[str, int]]]:
    upper_text = text.upper()
    vin_match = VIN_PATTERN.search(upper_text)
    year_match = YEAR_PATTERN.search(upper_text)
    title_match = TITLE_PATTERN.search(upper_text)
    state_match = STATE_PATTERN.search(upper_text)

    title_number = title_match.group(1).replace("-", "") if title_match else None
    if not title_number:
        # High-priority: 8-digit numeric title number (common US title format).
        # Apply OCR-noise correction (O→0, I→1) before searching so that
        # misreads like "5O312391" or "583I2391" can still match.
        noise_corrected = upper_text.translate(_OCR_NOISE_TABLE)
        eight_digit_match = TITLE_8DIGIT_PATTERN.search(noise_corrected)
        if eight_digit_match:
            title_number = eight_digit_match.group(1)
    if not title_number:
        excluded_tokens = {
            "VIN",
            "YEAR",
            "STATE",
            "TITLE",
            "NUMBER",
            "NO",
            "VEHICLE",
        }
        for token in re.findall(r"\b[A-Z0-9\-]{3,20}\b", upper_text):
            candidate = token.replace("-", "")
            if candidate in excluded_tokens:
                continue
            if VIN_PATTERN.fullmatch(candidate):
                continue
            if YEAR_PATTERN.fullmatch(candidate):
                continue
            if not any(character.isdigit() for character in candidate):
                continue
            title_number = candidate
            break

    return {
        "state": state_match.group(1) if state_match else None,
        "title_number": title_number,
        "vin": vin_match.group(1) if vin_match else None,
        "vehicle_year": int(year_match.group(1)) if year_match else None,
    }


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    document = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    try:
        for page in document:
            page_text = page.get_text("text").strip()
            if page_text:
                chunks.append(page_text)
                continue
            pixmap = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            # Apply preprocessing before OCR.
            try:
                image = preprocess_title_image(image)
            except Exception:
                pass
            image = binarize_image(image)
            chunks.append(pytesseract.image_to_string(image))
    finally:
        document.close()
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)

    # Full pre-processing pipeline (deskew, border crop, perspective correction).
    try:
        image = preprocess_title_image(image)
    except Exception:
        pass

    # Upscale to ensure the long side is at least 2000 px for OCR accuracy.
    width, height = image.size
    if max(width, height) < 2000:
        scale = 2000 / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )
    # Binarize for Tesseract.
    try:
        image = binarize_image(image)
    except Exception:
        image = image.convert("L")
        image = image.filter(ImageFilter.SHARPEN)
        image = ImageOps.autocontrast(image)
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    parts = filename.rsplit(".", 1)
    if len(parts) != 2:
        return ""
    extension = parts[1].lower()
    if extension == "pdf":
        return _extract_text_from_pdf(file_bytes)
    return _extract_text_from_image(file_bytes)


def _tesseract_fields_and_confidence(
    text: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run regex parsing on OCR text and assign a nominal confidence.

    Only the core fields are parseable from raw OCR text; extended fields
    (make, model, etc.) are left as None with 0.0 confidence.
    """
    core_fields = parse_extracted_fields(text) if text else {f: None for f in AI_CORE_FIELDS}
    fields: Dict[str, Any] = {f: None for f in _FIELD_NAMES}
    fields.update(core_fields)
    confidence = {
        f: (0.7 if fields.get(f) is not None else 0.0) for f in _FIELD_NAMES
    }
    return fields, confidence


def _get_word_confidences(file_bytes: bytes) -> Dict[str, float]:
    """Return per-word Tesseract confidence scores for an image file.

    Words are upper-cased; confidence is normalised to 0.0–1.0.
    Returns an empty dict on any error (e.g. PDF or corrupt input).
    """
    word_conf: Dict[str, float] = {}
    try:
        image = Image.open(io.BytesIO(file_bytes))
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        for word, conf in zip(data.get("text", []), data.get("conf", [])):
            word = (word or "").strip().upper()
            if word and conf != -1:
                norm = max(0.0, min(1.0, float(conf) / 100.0))
                if word not in word_conf or norm > word_conf[word]:
                    word_conf[word] = norm
    except Exception:
        pass
    return word_conf


def _apply_word_confidence(
    fields: Dict[str, Any],
    confidence: Dict[str, float],
    word_conf: Dict[str, float],
) -> None:
    """Override nominal confidence in-place with real Tesseract word scores.

    For each extracted field value, looks up the corresponding token in
    *word_conf* (Tesseract per-word confidence map) and replaces the
    nominal score when a match is found.
    """
    for field in _FIELD_NAMES:
        val = fields.get(field)
        if val is not None:
            key = str(val).upper()
            if key in word_conf:
                confidence[field] = word_conf[key]


def _detect_file_extension(file_bytes: bytes) -> str:
    """Detect a safe file extension from magic bytes.

    This avoids using user-provided filenames in the saved path, which
    eliminates taint flows from untrusted input into filesystem operations.
    """
    if file_bytes[:4] == b"%PDF":
        return "pdf"
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if file_bytes[:2] in (b"\xff\xd8",):
        return "jpg"
    if file_bytes[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "tif"
    if file_bytes[:2] in (b"BM",):
        return "bmp"
    return "bin"


def _save_upload(file_bytes: bytes, _original_filename: str) -> Optional[str]:
    """Persist an uploaded file to the uploads directory.

    The saved filename is a random UUID with an extension derived from the
    file's magic bytes — no user-supplied data appears in the saved path.

    Returns the saved file path, or None on failure.
    """
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        ext = _detect_file_extension(file_bytes)
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(UPLOADS_DIR, unique_name)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path
    except Exception:
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_extraction(
    filename: str,
    file_bytes: bytes,
    provider: str = EXTRACTION_PROVIDER,
    threshold: float = AI_CONFIDENCE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Run extraction using the configured provider.

    Barcode scanning always runs first; a successful decode provides
    1.0-confidence anchor values that are merged into the final result.

    Returns a dict with:
        fields          – all title fields (core + extended), None when absent
        raw_text        – OCR/extracted text for database storage
        source          – "barcode" | "ai" | "tesseract" | "hybrid"
        confidence      – per-field float confidence dict
        low_confidence  – list of field names below *threshold*

    Returns None when no text or data could be extracted at all.
    """
    ai_result = None
    raw_text: Optional[str] = None

    def _get_raw_text() -> str:
        nonlocal raw_text
        if raw_text is None:
            raw_text = extract_text_from_upload(filename, file_bytes)
        return raw_text

    # ── Barcode scan (highest confidence – runs regardless of provider) ───────
    barcode_result: Optional[Dict[str, Any]] = None
    try:
        barcode_result = scan_barcodes(filename, file_bytes)
    except Exception:
        barcode_result = None

    if provider in ("ai", "hybrid"):
        try:
            ai_result = extract_fields_with_ai(filename, file_bytes)
        except Exception:
            ai_result = None

    if provider == "tesseract":
        raw_text_value = _get_raw_text()
        if not raw_text_value:
            return None
        fields, confidence = _tesseract_fields_and_confidence(raw_text_value)
        # Override nominal confidence with real per-word Tesseract scores.
        word_conf = _get_word_confidences(file_bytes)
        if word_conf:
            _apply_word_confidence(fields, confidence, word_conf)
        source = "tesseract"
    else:
        # "ai" or "hybrid"
        if ai_result is None:
            raw_text_value = _get_raw_text()
            if not raw_text_value:
                return None
            # AI unavailable – fall back to Tesseract
            fields, confidence = _tesseract_fields_and_confidence(raw_text_value)
            # Override nominal confidence with real per-word Tesseract scores.
            word_conf = _get_word_confidences(file_bytes)
            if word_conf:
                _apply_word_confidence(fields, confidence, word_conf)
            source = "tesseract"
        elif provider == "ai":
            fields = {f: ai_result.get(f) for f in _FIELD_NAMES}
            confidence = ai_result.get("confidence", {f: 0.0 for f in _FIELD_NAMES})
            source = "ai"
        else:
            # hybrid: use AI where confident, Tesseract elsewhere (only if needed)
            ai_conf_map = ai_result.get("confidence", {})
            needs_tesseract = any(
                ai_result.get(field) is None or ai_conf_map.get(field, 0.0) < threshold
                for field in AI_CORE_FIELDS  # Only core fields can be found by Tesseract.
            )
            if needs_tesseract:
                raw_text_value = _get_raw_text()
                tesseract_fields, tesseract_conf = _tesseract_fields_and_confidence(
                    raw_text_value or ""
                )
                # Override nominal confidence with real per-word Tesseract scores.
                word_conf = _get_word_confidences(file_bytes)
                if word_conf:
                    _apply_word_confidence(tesseract_fields, tesseract_conf, word_conf)
            else:
                tesseract_fields = {f: None for f in _FIELD_NAMES}
                tesseract_conf = {f: 0.0 for f in _FIELD_NAMES}

            fields = {}
            confidence = {}
            ai_used = False
            tess_used = False
            for field in _FIELD_NAMES:
                ai_val = ai_result.get(field)
                ai_conf = ai_conf_map.get(field, 0.0)
                if ai_val is not None and ai_conf >= threshold:
                    fields[field] = ai_val
                    confidence[field] = ai_conf
                    ai_used = True
                else:
                    tess_val = tesseract_fields.get(field)
                    fields[field] = tess_val
                    confidence[field] = tesseract_conf.get(field, 0.0)
                    # Only mark tesseract as used when it actually produced a value.
                    if tess_val is not None:
                        tess_used = True
            if ai_used and tess_used:
                source = "hybrid"
            elif ai_used:
                source = "ai"
            else:
                source = "tesseract"

    # ── Merge barcode anchors (always override with 1.0-confidence values) ───
    if barcode_result:
        bc_conf = barcode_result.get("confidence", {})
        for field in _FIELD_NAMES:
            bc_val = barcode_result.get(field)
            if bc_val is not None:
                fields[field] = bc_val
                confidence[field] = bc_conf.get(field, 1.0)
        source = "barcode" if source == "tesseract" and barcode_result else source

    raw_text_out = raw_text or ""

    low_confidence: List[str] = [
        f for f in _FIELD_NAMES if confidence.get(f, 0.0) < threshold
    ]
    return {
        "fields": fields,
        "raw_text": raw_text_out,
        "source": source,
        "confidence": confidence,
        "low_confidence": low_confidence,
    }


def get_port_from_environment(default_port: int = DEFAULT_PORT) -> int:
    port_value = os.getenv("PORT", str(default_port)).strip()
    try:
        port = int(port_value)
    except ValueError:
        return default_port
    return port if 1 <= port <= 65535 else default_port


# ── LLM prompt-tuning ────────────────────────────────────────────────────────

# Canonical field order used by the Modelfile few-shot examples.
_LLM_FIELDS: Tuple[str, ...] = (
    "state", "title_number", "vin", "vehicle_year", "make", "model",
    "body_style", "color", "odometer", "owner_name", "owner_address",
    "purchase_price", "sale_date", "issue_date",
)

# Maximum number of few-shot MESSAGE pairs written into the Modelfile.
_LLM_MAX_EXAMPLES = 30


def _build_llm_modelfile(corrections: List[Dict[str, Any]], base_model: str) -> str:
    """Build an Ollama Modelfile incorporating ground-truth few-shot examples.

    The generated Modelfile:
    - Starts from the configured base model (``FROM <base_model>``).
    - Adds a detailed SYSTEM prompt covering all 14 NMVITIS fields.
    - Injects up to ``_LLM_MAX_EXAMPLES`` MESSAGE pairs (one per annotated
      title record) so the model learns the expected JSON output format from
      real ground-truth corrections.

    Parameters
    ----------
    corrections:
        List of dicts returned by ``list_ground_truth_corrections()``.
    base_model:
        Ollama model tag used as the base (e.g. ``"moondream"``).
    """
    from collections import defaultdict

    # Group corrections by record_id so each title becomes one example.
    by_record: Dict[int, Dict[str, str]] = defaultdict(dict)
    for corr in corrections:
        rid = int(corr["record_id"])
        field = corr.get("field_name", "")
        value = corr.get("corrected_value") or ""
        if field:
            by_record[rid][field] = value

    system_prompt = (
        "You are a vehicle title OCR assistant specializing in US motor vehicle "
        "title documents.\n\n"
        "Extract ALL of the following fields from the vehicle title image and "
        "return them as a single JSON object. Return null for any field that is "
        "not clearly visible or legible.\n\n"
        "Fields:\n"
        "- state: Two-letter US state abbreviation (e.g. NM, TX, CA).\n"
        "- title_number: Alphanumeric title number, 3-20 characters, no spaces.\n"
        "- vin: 17-character Vehicle Identification Number (uppercase, no I/O/Q).\n"
        "- vehicle_year: 4-digit model year as an integer.\n"
        "- make: Vehicle manufacturer (e.g. FORD, CHEVROLET, TOYOTA).\n"
        "- model: Vehicle model name (e.g. F-150, SILVERADO, CAMRY).\n"
        "- body_style: Body style code (e.g. SDN, PK, SUV, VAN).\n"
        "- color: Primary exterior color.\n"
        "- odometer: Integer odometer reading in miles.\n"
        "- owner_name: Full legal name of the registered owner.\n"
        "- owner_address: Full mailing address of the registered owner.\n"
        "- purchase_price: Purchase price as a numeric string (no currency symbol).\n"
        "- sale_date: Date of sale in YYYY-MM-DD format.\n"
        "- issue_date: Title issue date in YYYY-MM-DD format.\n\n"
        "Rules:\n"
        "- Return ONLY the JSON object with exactly these 14 keys.\n"
        "- vehicle_year and odometer must be integers or null.\n"
        "- All other fields must be strings or null.\n"
        "- Do not include markdown, explanations, or any other text."
    )

    user_msg = (
        "Extract all fields from this vehicle title image and return them as a JSON object."
    )

    lines: List[str] = [
        f"FROM {base_model}",
        "",
        f'SYSTEM """\n{system_prompt}\n"""',
    ]

    # Add few-shot MESSAGE pairs from ground-truth corrections.
    for labels in list(by_record.values())[:_LLM_MAX_EXAMPLES]:
        full_labels: Dict[str, Any] = {f: labels.get(f) for f in _LLM_FIELDS}
        # Strip triple-quotes from values to avoid breaking Modelfile syntax.
        for k, v in full_labels.items():
            if isinstance(v, str):
                full_labels[k] = v.replace('"""', "")
        assistant_msg = json.dumps(full_labels, ensure_ascii=False)
        lines.append("")
        lines.append(f'MESSAGE user "{user_msg}"')
        lines.append(f'MESSAGE assistant """\n{assistant_msg}\n"""')

    return "\n".join(lines)


def create_app(default_state: str = DEFAULT_STATE) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["DEFAULT_STATE"] = default_state.strip().upper()

    # ── Core endpoints ────────────────────────────────────────────────────────

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/status")
    def api_status():
        return jsonify(
            {
                "extraction_provider": EXTRACTION_PROVIDER,
                "ai_status": _check_ai_status(),
            }
        )

    @app.route("/diagnostics")
    def diagnostics():
        """Render the diagnostics page."""
        return render_template("diagnostics.html")

    @app.route("/help")
    def help_page():
        """Render the help / how-to documentation page."""
        return render_template("help.html")

    @app.route("/api/diagnostics/ai-progress")
    def api_ai_progress():
        """Return enriched AI status including pull-progress data from logs.

        Response keys:
            ai_status          – "ready" | "loading" | "unavailable" | "not_configured"
            extraction_provider – value of EXTRACTION_PROVIDER env var
            ai_model           – configured AI_MODEL value
            ai_endpoint        – configured AI_ENDPOINT value
            pull_progress      – progress dict (only populated when status == "loading"),
                                 or None otherwise
        """
        status = _check_ai_status()
        endpoint = os.getenv("AI_ENDPOINT", "http://ollama:11434")
        model = os.getenv("AI_MODEL", "moondream")
        result: Dict[str, Any] = {
            "ai_status": status,
            "extraction_provider": EXTRACTION_PROVIDER,
            "ai_model": model,
            "ai_endpoint": endpoint,
            "pull_progress": None,
        }
        if status == "loading":
            result["pull_progress"] = _get_ai_pull_progress()
        return jsonify(result)

    @app.route("/", methods=["GET", "POST"])
    def index():
        context: Dict[str, object] = {
            "message": None,
            "error": None,
            "record": None,
            "fields": None,
            "extraction_info": None,
        }
        if request.method == "POST":
            state_override = request.form.get("state", "").strip().upper()
            upload = request.files.get("file")
            if not upload or not upload.filename:
                context["error"] = "Please select a PDF or image file."
                return render_template("index.html", **context)
            if not allowed_file(upload.filename):
                context["error"] = "Unsupported file type."
                return render_template("index.html", **context)

            file_bytes = upload.read()
            if not file_bytes:
                context["error"] = "Uploaded file is empty."
                return render_template("index.html", **context)

            manual_entry = request.form.get("manual_entry") == "1"

            if manual_entry:
                state = state_override or app.config["DEFAULT_STATE"]
                source_file_path = _save_upload(file_bytes, upload.filename or "upload")
                connection = create_connection_from_env()
                try:
                    initialize_database(connection)
                    record = insert_title_record(
                        connection=connection,
                        state=state,
                        title_number="",
                        vin="",
                        vehicle_year=MISSING_YEAR_SENTINEL,
                        source_file_path=source_file_path,
                    )
                finally:
                    connection.close()
                return redirect(url_for("review_record", record_id=record["id"]))

            extraction = _run_extraction(upload.filename, file_bytes)
            if extraction is None:
                context["error"] = "No text could be extracted from the file."
                return render_template("index.html", **context)

            fields = extraction["fields"]
            state = state_override or fields.get("state") or app.config["DEFAULT_STATE"]
            title_number = fields.get("title_number") or ""
            vin = fields.get("vin") or ""
            vehicle_year = fields.get("vehicle_year") or MISSING_YEAR_SENTINEL
            extracted_text = extraction["raw_text"]

            # Save the uploaded file for the review workflow.
            source_file_path = _save_upload(file_bytes, upload.filename or "upload")

            connection = create_connection_from_env()
            try:
                initialize_database(connection)
                record = insert_title_record(
                    connection=connection,
                    state=state,
                    title_number=title_number,
                    vin=vin,
                    vehicle_year=vehicle_year,
                    ocr_text=extracted_text,
                    make=fields.get("make"),
                    model=fields.get("model"),
                    body_style=fields.get("body_style"),
                    color=fields.get("color"),
                    odometer=fields.get("odometer"),
                    owner_name=fields.get("owner_name"),
                    owner_address=fields.get("owner_address"),
                    purchase_price=fields.get("purchase_price"),
                    sale_date=fields.get("sale_date"),
                    issue_date=fields.get("issue_date"),
                    source_file_path=source_file_path,
                )
            finally:
                connection.close()

            context["fields"] = {
                "state": state,
                "title_number": title_number,
                "vin": vin,
                "vehicle_year": vehicle_year,
                "make": fields.get("make"),
                "model": fields.get("model"),
                "body_style": fields.get("body_style"),
                "color": fields.get("color"),
                "odometer": fields.get("odometer"),
                "owner_name": fields.get("owner_name"),
                "owner_address": fields.get("owner_address"),
                "purchase_price": fields.get("purchase_price"),
                "sale_date": fields.get("sale_date"),
                "issue_date": fields.get("issue_date"),
            }
            context["record"] = record
            context["extraction_info"] = {
                "source": extraction["source"],
                "confidence": extraction["confidence"],
                "low_confidence": extraction["low_confidence"],
            }
            if record["is_validated"]:
                context["message"] = "Record extracted, validated, and saved."
            else:
                context["error"] = "Record saved with validation errors."

        return render_template("index.html", **context)

    # ── Review / annotation endpoints ─────────────────────────────────────────

    @app.route("/review")
    def review_list():
        """List all title records for review."""
        try:
            limit = max(1, min(200, int(request.args.get("limit", "50"))))
            offset = max(0, int(request.args.get("offset", "0")))
        except (TypeError, ValueError):
            limit, offset = 50, 0
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            records = list_records(connection, limit=limit, offset=offset)
        finally:
            connection.close()
        return render_template("review.html", records=records, limit=limit, offset=offset)

    @app.route("/review/<int:record_id>", methods=["GET", "POST"])
    def review_record(record_id: int):
        """Show a single title record for inline correction."""
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            record = get_record_by_id(connection, record_id)
            if record is None:
                return jsonify({"error": "Record not found."}), 404

            if request.method == "POST":
                is_ground_truth = (
                    request.form.get("approve") == "1"
                    or request.form.get("bulk_approve") == "1"
                )
                corrections_saved = 0
                source_path = record.get("source_file_path") or ""
                corrected_fields: Dict[str, Any] = {}
                for field in _FIELD_NAMES:
                    corrected = request.form.get(f"field_{field}", "").strip()
                    if corrected:
                        insert_correction(
                            connection,
                            record_id=record_id,
                            field_name=field,
                            original_value=str(record.get(field) or ""),
                            corrected_value=corrected,
                            image_hash=source_path,
                            is_ground_truth=is_ground_truth,
                        )
                        corrected_fields[field] = corrected
                        corrections_saved += 1
                if corrected_fields:
                    update_title_record_fields(connection, record_id, corrected_fields)
                return jsonify({"saved": corrections_saved}), 200

            corrections = get_corrections_for_record(connection, record_id)
        finally:
            connection.close()

        # Build per-field GT status for the annotation progress bar.
        gt_fields = {c["field_name"] for c in corrections if c.get("is_ground_truth")}
        gt_count = len(gt_fields)

        return render_template(
            "review_record.html",
            record=record,
            corrections=corrections,
            field_names=list(_FIELD_NAMES),
            gt_fields=gt_fields,
            gt_count=gt_count,
            total_fields=len(_FIELD_NAMES),
        )

    # ── Export endpoint ───────────────────────────────────────────────────────

    @app.route("/export")
    def export_csv():
        """Download a NMVITIS-aligned CSV of validated records.

        Optional query parameters:
            date_from – YYYY-MM-DD  (inclusive lower bound on created_at)
            date_to   – YYYY-MM-DD  (inclusive upper bound on created_at)
        """
        import tempfile

        date_from = request.args.get("date_from", "").strip() or None
        date_to = request.args.get("date_to", "").strip() or None

        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp_path = tmp.name
            export_validated_to_csv_by_date(
                connection, tmp_path, date_from=date_from, date_to=date_to
            )
        finally:
            connection.close()

        return send_file(
            tmp_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name="nmvitis_export.csv",
        )

    # ── PDF generation endpoint ───────────────────────────────────────────────

    @app.route("/api/generate-pdf/<int:record_id>")
    def generate_pdf(record_id: int):
        """Generate and return the NMVITIS submission PDF for a record."""
        from pdf_generator import generate_nmvitis_pdf

        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            record = get_record_by_id(connection, record_id)
        finally:
            connection.close()

        if record is None:
            return jsonify({"error": "Record not found."}), 404

        try:
            pdf_path = generate_nmvitis_pdf(record, output_dir=PDFS_DIR)
        except Exception:
            return jsonify({"error": "PDF generation failed. Check server logs."}), 500

        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
        )

    # ── NMVITIS rejection import endpoint ────────────────────────────────────

    @app.route("/import-rejections", methods=["GET", "POST"])
    def import_rejections():
        """Accept a NMVITIS rejection CSV upload and flag affected records."""
        if request.method == "POST":
            upload = request.files.get("file")
            if not upload or not upload.filename:
                return jsonify({"error": "No file provided."}), 400
            import tempfile

            file_bytes = upload.read()
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="wb"
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            connection = create_connection_from_env()
            try:
                initialize_database(connection)
                count = import_nmvitis_rejections(connection, tmp_path)
            finally:
                connection.close()

            return jsonify({"corrections_created": count})

        return render_template("import_rejections.html")

    # ── Back-of-title endpoints ───────────────────────────────────────────────

    @app.route("/titles/<int:record_id>/back", methods=["GET", "POST"])
    def title_back(record_id: int):
        """Show or submit the back-of-title form for a record."""
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            front = get_record_by_id(connection, record_id)
            if front is None:
                return jsonify({"error": "Record not found."}), 404

            if request.method == "POST":
                try:
                    odometer_raw = request.form.get("odometer_at_sale", "").strip()
                    odometer_at_sale = int(odometer_raw) if odometer_raw.isdigit() else None
                    sale_price_raw = request.form.get("sale_price", "").strip()
                    sale_price = float(sale_price_raw) if sale_price_raw else None
                except (ValueError, TypeError):
                    odometer_at_sale = None
                    sale_price = None

                back_id = insert_title_back_record(
                    connection,
                    title_record_id=record_id,
                    odometer_at_sale=odometer_at_sale,
                    buyer_name=request.form.get("buyer_name", "").strip() or None,
                    buyer_address=request.form.get("buyer_address", "").strip() or None,
                    seller_name=request.form.get("seller_name", "").strip() or None,
                    sale_price=sale_price,
                    sale_date=request.form.get("sale_date", "").strip() or None,
                    notes=request.form.get("notes", "").strip() or None,
                )
                return jsonify({"id": back_id, "saved": True})

            back_record = get_title_back_record(connection, record_id)
        finally:
            connection.close()

        return render_template(
            "back_of_title.html",
            front=front,
            back=back_record,
        )

    # ── Uploaded file serving ─────────────────────────────────────────────────

    @app.route("/uploads/<filename>")
    def serve_upload(filename: str):
        """Serve a saved upload for display in the review UI.

        Only UUID-format filenames (generated by _save_upload) are accepted.
        Using Flask's send_from_directory ensures the response is confined to
        the uploads directory.
        """
        import re as _re

        if not _re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{1,8}", filename):
            return jsonify({"error": "File not found."}), 404
        return send_from_directory(UPLOADS_DIR, filename)

    # ── Training data export ──────────────────────────────────────────────────

    @app.route("/api/training-data")
    def training_data():
        """Return ground-truth corrections as JSON for training pipeline use."""
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            records = list_ground_truth_corrections(connection)
        finally:
            connection.close()
        return jsonify({"count": len(records), "corrections": records})

    # ── Annotation queue ──────────────────────────────────────────────────────

    @app.route("/annotate")
    def annotate_list():
        """Annotation queue – records prioritised by least GT coverage."""
        try:
            limit = max(1, min(200, int(request.args.get("limit", "50"))))
            offset = max(0, int(request.args.get("offset", "0")))
        except (TypeError, ValueError):
            limit, offset = 50, 0
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            records = get_annotation_queue(connection, limit=limit, offset=offset)
        finally:
            connection.close()
        return render_template(
            "annotate.html", records=records, limit=limit, offset=offset
        )

    # ── Training dashboard ────────────────────────────────────────────────────

    @app.route("/training")
    def training_dashboard():
        """Training dashboard showing GT dataset stats and run history."""
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            stats = get_training_stats(connection)
            runs = list_training_runs(connection)
        finally:
            connection.close()
        return render_template("training.html", stats=stats, runs=runs)

    @app.route("/api/training/stats")
    def api_training_stats():
        """Return ground-truth training dataset statistics as JSON."""
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            stats = get_training_stats(connection)
        finally:
            connection.close()
        return jsonify(stats)

    @app.route("/api/training/export", methods=["POST"])
    def api_training_export():
        """Export all ground-truth samples as a downloadable ZIP archive.

        Optionally accepts a ``notes`` form field to record with the run.
        """
        import io as _io
        import shutil as _shutil
        import tempfile as _tempfile
        import uuid as _uuid
        import zipfile as _zipfile
        from collections import defaultdict
        from pathlib import Path as _Path

        notes = request.form.get("notes", "").strip()

        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            corrections = list_ground_truth_corrections(connection)
        finally:
            connection.close()

        # Group corrections by record_id.
        by_record: Dict[int, list] = defaultdict(list)
        record_image: Dict[int, str] = {}
        for corr in corrections:
            rid = int(corr["record_id"])
            by_record[rid].append(corr)
            if corr.get("source_file_path"):
                record_image[rid] = corr["source_file_path"]

        tmp_dir = _Path(_tempfile.mkdtemp())
        samples_dir = tmp_dir / "samples"
        samples_dir.mkdir()

        manifest_samples = []
        exported = 0

        try:
            for record_id, corrs in by_record.items():
                sample_id = str(_uuid.uuid4())
                src_path = record_image.get(record_id)
                image_dest: Optional[str] = None
                image_ext = ""

                if src_path and _Path(src_path).exists():
                    image_ext = _Path(src_path).suffix or ".png"
                    image_dest = str(samples_dir / f"{sample_id}{image_ext}")
                    _shutil.copy2(src_path, image_dest)

                labels = {c["field_name"]: c["corrected_value"] for c in corrs}
                label_path = samples_dir / f"{sample_id}.json"
                label_path.write_text(
                    json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                manifest_samples.append(
                    {
                        "id": sample_id,
                        "record_id": record_id,
                        "image": f"samples/{sample_id}{image_ext}" if image_dest else None,
                        "labels": labels,
                        "fields": list(labels.keys()),
                    }
                )
                exported += 1

            manifest_path = tmp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"total": exported, "samples": manifest_samples},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # Build in-memory ZIP.
            zip_buf = _io.BytesIO()
            with _zipfile.ZipFile(zip_buf, "w", _zipfile.ZIP_DEFLATED) as zf:
                for file in tmp_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(tmp_dir))
            zip_buf.seek(0)
        finally:
            _shutil.rmtree(tmp_dir, ignore_errors=True)

        # Record the export.
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            insert_training_run(connection, sample_count=exported, notes=notes)
        finally:
            connection.close()

        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="training_data.zip",
        )

    @app.route("/api/training/apply-to-llm", methods=["POST"])
    def api_apply_to_llm():
        """Build a custom Ollama model from ground-truth corrections.

        Constructs a Modelfile with a detailed SYSTEM prompt and few-shot
        MESSAGE pairs derived from approved corrections, then POSTs it to
        the Ollama ``/api/create`` endpoint.  On success, records the event
        in ``training_runs`` and returns the new model name.
        """
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            corrections = list_ground_truth_corrections(connection)
        finally:
            connection.close()

        if not corrections:
            return (
                jsonify(
                    {
                        "error": (
                            "No ground-truth corrections found. "
                            "Annotate some records first, then try again."
                        )
                    }
                ),
                400,
            )

        custom_model = "titles-custom"
        # LLM_BASE_MODEL explicitly sets the Modelfile FROM base.  Fall back to
        # AI_MODEL, but guard against the circular case where AI_MODEL has
        # already been set to 'titles-custom' — using the custom model as its
        # own base causes Ollama to return HTTP 400.
        base_model = os.getenv("LLM_BASE_MODEL") or os.getenv("AI_MODEL", "moondream")
        if base_model == custom_model:
            return (
                jsonify(
                    {
                        "error": (
                            f"AI_MODEL is set to '{custom_model}', which cannot be used "
                            "as its own base model. "
                            "Add LLM_BASE_MODEL=<your-base-model> (e.g. LLM_BASE_MODEL=moondream) "
                            "to your .env file and restart the stack, then try again."
                        )
                    }
                ),
                400,
            )
        modelfile = _build_llm_modelfile(corrections, base_model)

        endpoint = os.getenv("AI_ENDPOINT", "http://ollama:11434")
        try:
            resp = requests.post(
                f"{endpoint}/api/create",
                json={"name": custom_model, "modelfile": modelfile, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return (
                jsonify(
                    {
                        "error": (
                            "Ollama endpoint is unavailable. "
                            "Ensure the AI service is running (COMPOSE_PROFILES=ai)."
                        )
                    }
                ),
                503,
            )
        except requests.exceptions.Timeout:
            return (
                jsonify({"error": "Ollama timed out while creating the custom model."}),
                504,
            )
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            detail = ""
            if exc.response is not None:
                try:
                    body = exc.response.json()
                    detail = ": " + (body.get("error") or str(body))
                except Exception:
                    detail = ": " + (exc.response.text or "")
            return (
                jsonify(
                    {
                        "error": (
                            f"Ollama returned HTTP {status_code} while creating the model{detail}"
                        )
                    }
                ),
                502,
            )

        unique_records = len({int(c["record_id"]) for c in corrections})
        notes = (
            f"LLM prompt-tuning: created '{custom_model}' "
            f"from '{base_model}' with {len(corrections)} corrections "
            f"across {unique_records} records"
        )
        connection = create_connection_from_env()
        try:
            initialize_database(connection)
            run_id = insert_training_run(
                connection, sample_count=unique_records, notes=notes
            )
        finally:
            connection.close()

        return jsonify(
            {
                "model": custom_model,
                "base_model": base_model,
                "corrections_used": len(corrections),
                "records_used": unique_records,
                "run_id": run_id,
                "message": (
                    f"Custom model '{custom_model}' created successfully from "
                    f"{unique_records} annotated title records. "
                    f"Set AI_MODEL={custom_model} in your .env and restart the stack to use it."
                ),
            }
        )

    # ── Maintenance endpoints ─────────────────────────────────────────────────

    @app.route("/api/maintenance/containers")
    def maintenance_containers():
        client = _docker_client()
        if client is None:
            return jsonify(
                {
                    "docker_available": False,
                    "services": {name: None for name in _MANAGED_SERVICES},
                    "error": "Docker socket not available.",
                }
            )
        return jsonify(
            {
                "docker_available": True,
                "services": _get_service_containers(client),
                "error": None,
            }
        )

    @app.route("/api/maintenance/logs/<service>")
    def maintenance_logs(service: str):
        if service not in _MANAGED_SERVICES:
            return jsonify({"error": "Unknown service.", "lines": []}), 400
        client = _docker_client()
        if client is None:
            return (
                jsonify({"error": "Docker socket not available.", "lines": []}),
                503,
            )
        try:
            lines = max(1, min(500, int(request.args.get("lines", "100"))))
        except (TypeError, ValueError):
            lines = 100
        result = _get_service_logs(client, service, lines)
        result["service"] = service
        return jsonify(result)

    @app.route("/api/maintenance/ai/start", methods=["POST"])
    def maintenance_start_ai():
        client = _docker_client()
        if client is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "messages": [],
                        "errors": [
                            "Docker socket not available. "
                            "Set COMPOSE_PROFILES=ai and redeploy the stack."
                        ],
                    }
                ),
                503,
            )
        # Invalidate the AI status cache so the next /api/status poll reflects reality.
        _AI_STATUS_CACHE["checked_at"] = 0.0
        result = _start_ai_services(client)
        return jsonify(result), (200 if result["success"] else 500)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=get_port_from_environment())
