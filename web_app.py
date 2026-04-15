import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import pytesseract
from flask import Flask, render_template, request
from PIL import Image

from ai_extraction import extract_fields_with_ai
from title_entry_tool import (
    create_connection_from_env,
    initialize_database,
    insert_title_record,
)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
DEFAULT_STATE = "NM"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MISSING_YEAR_SENTINEL = 0
DEFAULT_PORT = 8000

# Extraction provider: "tesseract" | "ai" | "hybrid" (default)
EXTRACTION_PROVIDER = os.getenv("EXTRACTION_PROVIDER", "hybrid").lower().strip()
try:
    AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.6"))
except ValueError:
    AI_CONFIDENCE_THRESHOLD = 0.6

_FIELD_NAMES: Tuple[str, ...] = ("state", "title_number", "vin", "vehicle_year")

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
YEAR_PATTERN = re.compile(r"\b(18[8-9]\d|19\d{2}|20\d{2}|21\d{2})\b")
TITLE_PATTERN = re.compile(
    r"\bTITLE\s*(?:NO|NUMBER|#)?\s*[:\-]?\s*([A-Z0-9\-]{3,20})\b",
    re.IGNORECASE,
)
STATE_PATTERN = re.compile(r"\bSTATE\s*[:\-]?\s*([A-Z]{2})\b", re.IGNORECASE)


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
            chunks.append(pytesseract.image_to_string(image))
    finally:
        document.close()
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image).strip()


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
    """Run regex parsing on OCR text and assign a nominal confidence."""
    fields = parse_extracted_fields(text) if text else {f: None for f in _FIELD_NAMES}
    confidence = {f: (0.7 if fields.get(f) is not None else 0.0) for f in _FIELD_NAMES}
    return fields, confidence


def _run_extraction(
    filename: str,
    file_bytes: bytes,
    provider: str = EXTRACTION_PROVIDER,
    threshold: float = AI_CONFIDENCE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Run extraction using the configured provider.

    Returns a dict with:
        fields          – {state, title_number, vin, vehicle_year}
        raw_text        – OCR/extracted text for database storage
        source          – "ai" | "tesseract" | "hybrid" | "hybrid-fallback"
        confidence      – per-field float confidence dict
        low_confidence  – list of field names below *threshold*

    Returns None when no text or data could be extracted at all.
    """
    raw_text = extract_text_from_upload(filename, file_bytes)

    ai_result = None
    if provider in ("ai", "hybrid"):
        try:
            ai_result = extract_fields_with_ai(filename, file_bytes)
        except Exception:
            ai_result = None

    if provider == "tesseract":
        if not raw_text:
            return None
        fields, confidence = _tesseract_fields_and_confidence(raw_text)
        source = "tesseract"
    else:
        # "ai" or "hybrid"
        if ai_result is None and not raw_text:
            return None

        if ai_result is None:
            # AI unavailable – fall back to Tesseract
            fields, confidence = _tesseract_fields_and_confidence(raw_text)
            source = "tesseract"
        elif provider == "ai":
            fields = {f: ai_result.get(f) for f in _FIELD_NAMES}
            confidence = ai_result.get("confidence", {f: 0.0 for f in _FIELD_NAMES})
            source = "ai"
        else:
            # hybrid: use AI where confident, Tesseract elsewhere
            tesseract_fields, tesseract_conf = _tesseract_fields_and_confidence(raw_text or "")
            fields = {}
            confidence = {}
            ai_used = False
            tess_used = False
            ai_conf_map = ai_result.get("confidence", {})
            for field in _FIELD_NAMES:
                ai_val = ai_result.get(field)
                ai_conf = ai_conf_map.get(field, 0.0)
                if ai_val is not None and ai_conf >= threshold:
                    fields[field] = ai_val
                    confidence[field] = ai_conf
                    ai_used = True
                else:
                    fields[field] = tesseract_fields.get(field)
                    confidence[field] = tesseract_conf.get(field, 0.0)
                    tess_used = True
            if ai_used and tess_used:
                source = "hybrid"
            elif ai_used:
                source = "ai"
            else:
                source = "tesseract"

    low_confidence: List[str] = [
        f for f in _FIELD_NAMES if confidence.get(f, 0.0) < threshold
    ]
    return {
        "fields": fields,
        "raw_text": raw_text or "",
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


def create_app(default_state: str = DEFAULT_STATE) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["DEFAULT_STATE"] = default_state.strip().upper()

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
                )
            finally:
                connection.close()

            context["fields"] = {
                "state": state,
                "title_number": title_number,
                "vin": vin,
                "vehicle_year": vehicle_year,
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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=get_port_from_environment())
