"""Barcode scanning for vehicle title documents.

Attempts to decode PDF417, QR, DataMatrix, and other 1-D/2-D barcodes
that are frequently printed on modern US state vehicle titles.  A
successful decode provides 1.0-confidence values that bypass AI and
Tesseract verification entirely.
"""

import io
import re
from typing import Any, Dict, List, Optional

import fitz
from PIL import Image

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyzbar.pyzbar import ZBarSymbol
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

BarcodeResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# VIN sanity pattern (no I, O, Q; exactly 17 chars)
# ---------------------------------------------------------------------------

_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

# Common PDF417 data separator characters used on US titles.
_AAMVA_SEPARATOR = "\x1e"
_ALT_SEPARATORS = ["|", ",", "\t"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _images_from_bytes(filename: str, file_bytes: bytes) -> List[Image.Image]:
    """Return a list of PIL images for all pages/frames in the uploaded file."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        images: List[Image.Image] = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in doc:
                pixmap = page.get_pixmap(dpi=200)
                images.append(Image.open(io.BytesIO(pixmap.tobytes("png"))))
        finally:
            doc.close()
        return images
    return [Image.open(io.BytesIO(file_bytes))]


def _parse_aamva_pdf417(data: str) -> BarcodeResult:
    """Parse an AAMVA-format PDF417 barcode (used on US driver licenses and
    some title barcodes).

    Returns a dict with whatever fields could be extracted.
    """
    fields: BarcodeResult = {}

    # AAMVA header starts with "@\n\x1e\rANSI " or similar.
    # For vehicle titles the encoding is less standardised; we do a best-effort
    # scan for known field codes.
    vin_match = re.search(r"\bDAQ([A-HJ-NPR-Z0-9]{17})\b", data)
    if vin_match:
        fields["vin"] = vin_match.group(1)

    return fields


def _parse_generic_barcode(raw: str) -> BarcodeResult:
    """Best-effort extraction from a non-AAMVA barcode payload.

    Looks for a 17-character VIN anywhere in the string.
    """
    fields: BarcodeResult = {}

    # Try to find a VIN-like sequence
    for candidate in re.findall(r"[A-HJ-NPR-Z0-9]{17}", raw.upper()):
        if _VIN_RE.match(candidate):
            fields["vin"] = candidate
            break

    # Some barcodes encode state+title as "NM:12345678" or "NM 12345678"
    state_title_match = re.search(
        r"\b([A-Z]{2})[:\- ]([A-Z0-9]{3,20})\b", raw.upper()
    )
    if state_title_match:
        fields["state"] = state_title_match.group(1)
        fields["title_number"] = state_title_match.group(2)

    # Year: look for a standalone 4-digit year
    year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", raw)
    if year_match:
        fields["vehicle_year"] = int(year_match.group(1))

    return fields


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan_barcodes(filename: str, file_bytes: bytes) -> Optional[BarcodeResult]:
    """Scan all pages of a title document for barcodes and return extracted
    field values.

    Returns a dict containing any decoded fields with a ``confidence`` sub-dict
    of 1.0 for successfully decoded values, or ``None`` if no barcode was found
    or pyzbar is not installed.

    Supported symbologies (via pyzbar / ZBar):
        PDF417, QR Code, DataMatrix, Code 128, Code 39, and others.
    """
    if not _PYZBAR_AVAILABLE:
        return None

    try:
        images = _images_from_bytes(filename, file_bytes)
    except Exception:
        return None

    merged: BarcodeResult = {}

    for image in images:
        try:
            decoded_barcodes = pyzbar_decode(image)
        except Exception:
            continue

        for barcode in decoded_barcodes:
            try:
                raw = barcode.data.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not raw:
                continue

            # Choose parser based on symbology and content.
            if barcode.type == "PDF417" and "ANSI" in raw:
                fields = _parse_aamva_pdf417(raw)
            else:
                fields = _parse_generic_barcode(raw)

            for key, val in fields.items():
                if val is not None and key not in merged:
                    merged[key] = val

    if not merged:
        return None

    # Build confidence dict — barcode-decoded values are always 1.0.
    conf: Dict[str, float] = {k: 1.0 for k in merged}
    merged["confidence"] = conf
    return merged
