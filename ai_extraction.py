"""Self-hosted AI vision extraction provider.

Calls an Ollama-compatible endpoint to extract title fields from images
using a multimodal LLM.  Falls back gracefully when the service is
unavailable or returns unrecognisable output.

Extraction uses a two-pass strategy:
  Pass 1 – full image + default crops → detect state and title_number as
            anchors, then load the matching state layout template.
  Pass 2 – full image + state-specific targeted crops → extract the
            complete field set using a state-aware prompt.
"""

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
_MAX_IMAGE_DIMENSION = 2000

# All fields extracted by the AI (core + extended).
_CORE_FIELDS = ("state", "title_number", "vin", "vehicle_year")
_EXTENDED_FIELDS = (
    "make", "model", "body_style", "color", "odometer",
    "owner_name", "owner_address", "purchase_price", "sale_date", "issue_date",
)
_ALL_FIELDS: Tuple[str, ...] = _CORE_FIELDS + _EXTENDED_FIELDS

# ---------------------------------------------------------------------------
# State template loader
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "state_templates"
_template_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def _load_state_template(state: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the state layout template dict, or None when unavailable."""
    if not state:
        return None
    key = state.upper()
    if key in _template_cache:
        return _template_cache[key]
    path = _TEMPLATE_DIR / f"{key}.json"
    if not path.exists():
        _template_cache[key] = None
        return None
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
        _template_cache[key] = template
        return template
    except (json.JSONDecodeError, OSError):
        _template_cache[key] = None
        return None

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

# Pass-1 prompt: extract anchors only (state + title_number).
_ANCHOR_PROMPT = (
    "You are a vehicle title OCR assistant. "
    "Look at this vehicle title document image.\n\n"
    "Return ONLY a JSON object (no markdown, no explanation):\n"
    '{\n'
    '  "state": "<2-letter US state abbreviation or null>",\n'
    '  "title_number": "<alphanumeric title number or null>",\n'
    '  "confidence": {"state": <0.0-1.0>, "title_number": <0.0-1.0>}\n'
    "}\n\n"
    "Rules:\n"
    "- State is a 2-letter abbreviation such as NM, TX, CA.\n"
    "- Title number is 3-20 alphanumeric characters.\n"
    "- Set a field to null and confidence to 0.0 when not found."
)

# Pass-2 full prompt (state-agnostic baseline).
_FULL_PROMPT_TEMPLATE = (
    "You are a vehicle title OCR assistant. "
    "Extract ALL of the following fields from this {state_hint}vehicle title "
    "document image.\n\n"
    "Return ONLY a JSON object with this exact structure "
    "(no markdown, no explanation):\n"
    "{{\n"
    '  "state": "<2-letter US state abbreviation or null>",\n'
    '  "title_number": "<alphanumeric title number, no spaces, or null>",\n'
    '  "vin": "<17-character vehicle identification number, or null>",\n'
    '  "vehicle_year": <4-digit year as integer, or null>,\n'
    '  "make": "<vehicle manufacturer name or null>",\n'
    '  "model": "<vehicle model name or null>",\n'
    '  "body_style": "<body style e.g. SDN/SEDAN/PU/PICKUP/SUV/VAN or null>",\n'
    '  "color": "<primary color or null>",\n'
    '  "odometer": <integer odometer reading in miles, or null>,\n'
    '  "owner_name": "<registered owner full name or null>",\n'
    '  "owner_address": "<registered owner address or null>",\n'
    '  "purchase_price": "<purchase price as string with no currency symbol, or null>",\n'
    '  "sale_date": "<date of sale as YYYY-MM-DD or null>",\n'
    '  "issue_date": "<title issue date as YYYY-MM-DD or null>",\n'
    '  "confidence": {{\n'
    '    "state": <0.0-1.0>, "title_number": <0.0-1.0>,\n'
    '    "vin": <0.0-1.0>, "vehicle_year": <0.0-1.0>,\n'
    '    "make": <0.0-1.0>, "model": <0.0-1.0>,\n'
    '    "body_style": <0.0-1.0>, "color": <0.0-1.0>,\n'
    '    "odometer": <0.0-1.0>, "owner_name": <0.0-1.0>,\n'
    '    "owner_address": <0.0-1.0>, "purchase_price": <0.0-1.0>,\n'
    '    "sale_date": <0.0-1.0>, "issue_date": <0.0-1.0>\n'
    '  }}\n'
    "}}\n\n"
    "Rules:\n"
    "- VIN is exactly 17 uppercase characters (no I, O, Q) and digits.\n"
    "- Title number is 3-20 alphanumeric characters, no spaces.\n"
    "- State is a 2-letter abbreviation.\n"
    "- vehicle_year is an integer between 1886 and 2100.\n"
    "- odometer is an integer (miles); strip commas before converting.\n"
    "- Dates must be formatted YYYY-MM-DD; use null if unknown.\n"
    "- Set any field to null and its confidence to 0.0 when not found.\n"
    "- Confidence 1.0 means certain; 0.0 means absent.{field_hints}"
)


def _build_full_prompt(state: Optional[str], template: Optional[Dict[str, Any]]) -> str:
    """Build a state-aware full-extraction prompt."""
    state_hint = f"{state} " if state else ""
    field_hints = ""
    if template and state:
        label_lines = []
        for field, fdata in template.get("fields", {}).items():
            hints = fdata.get("label_hints", [])
            if hints:
                label_lines.append(f'- "{field}" is labelled: {", ".join(hints[:3])}')
        if label_lines:
            field_hints = "\n\nField label hints for {state}:\n".format(state=state)
            field_hints += "\n".join(label_lines)
    return _FULL_PROMPT_TEMPLATE.format(
        state_hint=state_hint, field_hints=field_hints
    )


_VIN_PATTERN = re.compile(r"[^A-HJ-NPR-Z0-9]")

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _normalize_image(image: Image.Image, max_dimension: int = _MAX_IMAGE_DIMENSION) -> Image.Image:
    """Correct orientation, resize, sharpen, and enhance contrast for better extraction."""
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
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


def _image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base-64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_page_images_from_pdf(file_bytes: bytes) -> List[Image.Image]:
    """Render each PDF page to a PIL image at 300 dpi."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[Image.Image] = []
    try:
        for page in document:
            pixmap = page.get_pixmap(dpi=300)
            images.append(Image.open(io.BytesIO(pixmap.tobytes("png"))))
    finally:
        document.close()
    return images


def _get_image_from_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))


def _crop_field_regions(
    image: Image.Image, state: Optional[str] = None
) -> List[Image.Image]:
    """Return sub-image crops for title field locations.

    When a state template is available, returns one crop per field region
    defined in the template, providing targeted extraction zones for the AI.

    When no template is found (unknown state or state=None), falls back to
    two generic crops:
    - Top strip (first ~12 % of height, full width)  → VIN and vehicle year
    - Top-right corner (~15 % of width, ~10 % of height) → title number

    These targeted crops reduce background noise for the AI model when the
    full-page image is complex (security guilloché patterns, photos taken at
    an angle, etc.).
    """
    width, height = image.size
    template = _load_state_template(state)

    if template:
        crops: List[Image.Image] = []
        for field_data in template.get("fields", {}).values():
            for region in field_data.get("regions", []):
                left = max(0, int(region["left"] * width))
                top = max(0, int(region["top"] * height))
                right = min(width, int(region["right"] * width))
                bottom = min(height, int(region["bottom"] * height))
                if right > left and bottom > top:
                    crops.append(image.crop((left, top, right, bottom)))
        if crops:
            return crops

    # Default: two generic crops (backward-compatible behaviour).
    default_crops: List[Image.Image] = []

    # Top strip: full width × first 12 % of height (VIN / year row)
    top_strip_height = max(1, int(height * 0.12))
    default_crops.append(image.crop((0, 0, width, top_strip_height)))

    # Top-right corner: rightmost 15 % of width × first 10 % of height (title number)
    top_right_left = max(0, int(width * 0.85))
    top_right_height = max(1, int(height * 0.10))
    default_crops.append(image.crop((top_right_left, 0, width, top_right_height)))

    return default_crops


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


def _call_ollama(
    image_b64: str,
    endpoint: str,
    model: str,
    timeout: int,
    prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """POST one image to the Ollama /api/generate endpoint.

    Returns the parsed JSON body of the model response, or None on any
    network or parsing failure.
    """
    payload = {
        "model": model,
        "prompt": prompt or _build_full_prompt(None, None),
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

    # --- extended text fields (make, model, body_style, color, owner_name,
    #     owner_address, purchase_price, sale_date, issue_date) ---
    def _clean_text(field: str, max_len: int = 200) -> Optional[str]:
        val = raw.get(field)
        if not isinstance(val, str):
            return None
        cleaned = val.strip()
        return cleaned[:max_len] if cleaned else None

    make = _clean_text("make", 60)
    model = _clean_text("model", 60)
    body_style = _clean_text("body_style", 30)
    color = _clean_text("color", 30)
    owner_name = _clean_text("owner_name", 120)
    owner_address = _clean_text("owner_address", 200)
    purchase_price = _clean_text("purchase_price", 30)
    sale_date = _clean_text("sale_date", 20)
    issue_date = _clean_text("issue_date", 20)

    # --- odometer ---
    odometer_raw = raw.get("odometer")
    odometer: Optional[int] = None
    if isinstance(odometer_raw, (int, float)):
        odometer = int(odometer_raw) if odometer_raw >= 0 else None
    elif isinstance(odometer_raw, str):
        digits = re.sub(r"[^0-9]", "", odometer_raw)
        odometer = int(digits) if digits else None

    conf: Dict[str, float] = {f: _clamp_conf(f) for f in _ALL_FIELDS}

    return {
        "state": state,
        "title_number": title_number,
        "vin": vin,
        "vehicle_year": vehicle_year,
        "make": make,
        "model": model,
        "body_style": body_style,
        "color": color,
        "odometer": odometer,
        "owner_name": owner_name,
        "owner_address": owner_address,
        "purchase_price": purchase_price,
        "sale_date": sale_date,
        "issue_date": issue_date,
        "confidence": conf,
    }


# ---------------------------------------------------------------------------
# Multi-page merge
# ---------------------------------------------------------------------------


def _merge_page_results(results: List[ExtractionResult]) -> ExtractionResult:
    """Merge per-page results, keeping the highest-confidence value per field."""
    merged: ExtractionResult = {f: None for f in _ALL_FIELDS}
    merged["confidence"] = {f: 0.0 for f in _ALL_FIELDS}

    for result in results:
        conf = result.get("confidence", {})
        for field in _ALL_FIELDS:
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

    Uses a two-pass strategy:
      Pass 1 – sends the full image with an anchor prompt to quickly detect
               the state and title number.  The detected state is used to load
               a layout template and build a state-aware prompt for pass 2.
      Pass 2 – sends the full image plus state-specific crops with the
               complete field prompt.  Per-page results are merged by keeping
               the highest-confidence value for each field.

    Returns an ExtractionResult dict containing all fields and per-field
    confidence scores, or None if the model is unreachable or all calls fail.
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

    # ── Pass 1: anchor detection ───────────────────────────────────────────
    detected_state: Optional[str] = None
    for image in page_images:
        normalised = _normalize_image(image)
        image_b64 = _image_to_base64_png(normalised)
        anchor_raw = _call_ollama(image_b64, endpoint, model, timeout_int, _ANCHOR_PROMPT)
        anchor = _parse_ai_response(anchor_raw)
        if anchor and anchor.get("state"):
            detected_state = anchor["state"]
            break  # State found – no need to check more pages.

    template = _load_state_template(detected_state)
    full_prompt = _build_full_prompt(detected_state, template)

    # ── Pass 2: full extraction with state-aware prompt and crops ──────────
    page_results: List[ExtractionResult] = []
    for image in page_images:
        normalised = _normalize_image(image)
        image_b64 = _image_to_base64_png(normalised)
        raw = _call_ollama(image_b64, endpoint, model, timeout_int, full_prompt)
        parsed = _parse_ai_response(raw)
        if parsed is not None:
            page_results.append(parsed)

        # Send targeted field-region crops to the AI to reduce noise.
        for crop in _crop_field_regions(image, detected_state):
            normalised_crop = _normalize_image(crop)
            crop_b64 = _image_to_base64_png(normalised_crop)
            raw_crop = _call_ollama(crop_b64, endpoint, model, timeout_int, full_prompt)
            parsed_crop = _parse_ai_response(raw_crop)
            if parsed_crop is not None:
                page_results.append(parsed_crop)

    if not page_results:
        return None

    return _merge_page_results(page_results)
