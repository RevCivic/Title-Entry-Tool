import io
import re
from typing import Dict, Optional, Union

import fitz
import pytesseract
from flask import Flask, render_template, request
from PIL import Image

from title_entry_tool import (
    create_connection,
    initialize_database,
    insert_title_record,
)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
DEFAULT_STATE = "NM"

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
    extension = filename.rsplit(".", 1)[1].lower()
    if extension == "pdf":
        return _extract_text_from_pdf(file_bytes)
    return _extract_text_from_image(file_bytes)


def create_app(db_path: str = "titles.db", default_state: str = DEFAULT_STATE) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["DB_PATH"] = db_path
    app.config["DEFAULT_STATE"] = default_state.strip().upper()

    @app.route("/", methods=["GET", "POST"])
    def index():
        context: Dict[str, object] = {
            "message": None,
            "error": None,
            "record": None,
            "fields": None,
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

            extracted_text = extract_text_from_upload(upload.filename, file_bytes)
            if not extracted_text:
                context["error"] = "No text could be extracted from the file."
                return render_template("index.html", **context)

            fields = parse_extracted_fields(extracted_text)
            state = state_override or fields["state"] or app.config["DEFAULT_STATE"]
            title_number = fields["title_number"] or ""
            vin = fields["vin"] or ""
            vehicle_year = fields["vehicle_year"] or 0

            connection = create_connection(app.config["DB_PATH"])
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
            if record["is_validated"]:
                context["message"] = "Record extracted, validated, and saved."
            else:
                context["error"] = "Record saved with validation errors."

        return render_template("index.html", **context)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
