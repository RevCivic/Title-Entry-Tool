"""Self-hosted AI vision extraction provider.

Calls an Ollama-compatible endpoint to extract title fields from images
using a multimodal LLM.  Falls back gracefully when the service is
unavailable or returns unrecognisable output.
"""

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import fitz
import requests
from PIL import Image, ImageEnhance, ImageOps

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ExtractionResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# Defaults (overridden by environment variables at call time)
# ---------------------------------------------------------------------------

DEFAULT_AI_ENDPOINT = "http://ollama:11434"
DEFAULT_AI_MODEL = "moondream"
DEFAULT_AI_TIMEOUT = 60
DEFAULT_AI_CONFIDENCE_THRESHOLD = 0.6

# Maximum image dimension sent to the model (pixels on the longer side).
_MAX_IMAGE_DIMENSION = 1200

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = (
    "You are a document data extraction assistant. "
    "Extract the following fields from this vehicle title document image.\n\n"
    "Return ONLY a JSON object with this exact structure "
    "(no markdown, no explanation):\n"
    '{\n'
    '  "state": "<2-letter US state abbreviation or null>",\n'
    '  "title_number": "<alphanumeric title number, no spaces, or null>",\n'
    '  "vin": "<17-character vehicle identification number, or null>",\n'
    '  "vehicle_year": <4-digit year as integer, or null>,\n'
    '  "confidence": {\n'
    '    "state": <0.0-1.0>,\n'
    '    "title_number": <0.0-1.0>,\n'
    '    "vin": <0.0-1.0>,\n'
    '    "vehicle_year": <0.0-1.0>\n'
    '  }\n'
    "}\n\n"
    "Rules:\n"
    "- VIN is exactly 17 uppercase characters (no I, O, Q) and digits.\n"
    "- Title number is 3–20 alphanumeric characters.\n"
    "- State is a 2-letter abbreviation such as NM, TX, or CA.\n"
    "- vehicle_year is an integer between 1886 and 2100.\n"
    "- Set a field to null and its confidence to 0.0 when not found.\n"
    "- Confidence 1.0 means certain; 0.0 means absent."
)

_VIN_PATTERN = re.compile(r"[^A-HJ-NPR-Z0-9]")

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _normalize_image(image: Image.Image, max_dimension: int = _MAX_IMAGE_DIMENSION) -> Image.Image:
    """Correct orientation, resize, and enhance contrast for better extraction."""
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


def _image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base-64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_page_images_from_pdf(file_bytes: bytes) -> List[Image.Image]:
    """Render each PDF page to a PIL image at 150 dpi."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[Image.Image] = []
    try:
        for page in document:
            pixmap = page.get_pixmap(dpi=150)
            images.append(Image.open(io.BytesIO(pixmap.tobytes("png"))))
    finally:
        document.close()
    return images


def _get_image_from_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


def _call_ollama(
    image_b64: str,
    endpoint: str,
    model: str,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    """POST one image to the Ollama /api/generate endpoint.

    Returns the parsed JSON body of the model response, or None on any
    network or parsing failure.
    """
    payload = {
        "model": model,
        "prompt": _EXTRACTION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(
            f"{endpoint}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "")
        return json.loads(raw_response)
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_ai_response(raw: Optional[Any]) -> Optional[ExtractionResult]:
    """Validate and normalise a raw AI JSON response.

    Returns a standardised ExtractionResult or None if the response is
    too malformed to use.
    """
    if not raw or not isinstance(raw, dict):
        return None

    confidence_raw = raw.get("confidence", {})
    if not isinstance(confidence_raw, dict):
        confidence_raw = {}

    def _clamp_conf(field: str) -> float:
        val = confidence_raw.get(field, 0.0)
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return 0.0

    # --- state ---
    state = raw.get("state")
    if isinstance(state, str):
        state = re.sub(r"[^A-Z]", "", state.upper())[:2] or None
    else:
        state = None

    # --- title_number ---
    title_number = raw.get("title_number")
    if isinstance(title_number, str):
        title_number = re.sub(r"[^A-Z0-9]", "", title_number.upper())[:20] or None
    else:
        title_number = None

    # --- vin ---
    vin = raw.get("vin")
    if isinstance(vin, str):
        vin = _VIN_PATTERN.sub("", vin.upper())
        vin = vin if len(vin) == 17 else None
    else:
        vin = None

    # --- vehicle_year ---
    vehicle_year = raw.get("vehicle_year")
    if isinstance(vehicle_year, (int, float)):
        year_int = int(vehicle_year)
        vehicle_year = year_int if 1886 <= year_int <= 2100 else None
    else:
        vehicle_year = None

    return {
        "state": state,
        "title_number": title_number,
        "vin": vin,
        "vehicle_year": vehicle_year,
        "confidence": {
            "state": _clamp_conf("state"),
            "title_number": _clamp_conf("title_number"),
            "vin": _clamp_conf("vin"),
            "vehicle_year": _clamp_conf("vehicle_year"),
        },
    }


# ---------------------------------------------------------------------------
# Multi-page merge
# ---------------------------------------------------------------------------


def _merge_page_results(results: List[ExtractionResult]) -> ExtractionResult:
    """Merge per-page results, keeping the highest-confidence value per field."""
    merged: ExtractionResult = {
        "state": None,
        "title_number": None,
        "vin": None,
        "vehicle_year": None,
        "confidence": {
            "state": 0.0,
            "title_number": 0.0,
            "vin": 0.0,
            "vehicle_year": 0.0,
        },
    }
    for result in results:
        conf = result.get("confidence", {})
        for field in ("state", "title_number", "vin", "vehicle_year"):
            new_val = result.get(field)
            new_conf = conf.get(field, 0.0)
            if new_val is not None and new_conf > merged["confidence"][field]:
                merged[field] = new_val
                merged["confidence"][field] = new_conf
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_fields_with_ai(
    filename: str,
    file_bytes: bytes,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Optional[ExtractionResult]:
    """Extract title fields from an uploaded file using a self-hosted AI model.

    Each page / image is normalised, encoded, and sent to the configured
    Ollama endpoint.  Per-page results are merged by taking the
    highest-confidence value for each field.

    Returns an ExtractionResult dict containing:
        state, title_number, vin, vehicle_year  – extracted values (or None)
        confidence                               – per-field float 0.0–1.0

    Returns None if the model is unreachable or all page calls fail.
    """
    endpoint = endpoint or os.getenv("AI_ENDPOINT", DEFAULT_AI_ENDPOINT)
    model = model or os.getenv("AI_MODEL", DEFAULT_AI_MODEL)
    try:
        timeout_int = int(timeout if timeout is not None else os.getenv("AI_TIMEOUT", str(DEFAULT_AI_TIMEOUT)))
    except (TypeError, ValueError):
        timeout_int = DEFAULT_AI_TIMEOUT

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        page_images = _get_page_images_from_pdf(file_bytes)
    else:
        page_images = [_get_image_from_bytes(file_bytes)]

    page_results: List[ExtractionResult] = []
    for image in page_images:
        normalised = _normalize_image(image)
        image_b64 = _image_to_base64_png(normalised)
        raw = _call_ollama(image_b64, endpoint, model, timeout_int)
        parsed = _parse_ai_response(raw)
        if parsed is not None:
            page_results.append(parsed)

    if not page_results:
        return None

    return _merge_page_results(page_results)
