"""NMVITIS PDF submission form generator.

Produces a single-page PDF for each title record that can be submitted
to NMVITIS for record-keeping.  Files are named
``{state}_{title_number}_{vin}.pdf`` and saved to the configured data
directory.

Also provides :func:`generate_mv7_pdf` which fills the Pennsylvania MV-7
Scrap/Salvage Certificate form (``BLANK-MV7-FORM.pdf``) for a single record.
"""

import os
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = os.getenv("DATA_DIR", "/app/data")
_PDF_DIR = os.path.join(_DATA_DIR, "pdfs")

# Path to the blank MV-7 form shipped with the application.
_MV7_BLANK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "BLANK-MV7-FORM.pdf"
)

# NMVITIS field display labels in submission order.
_FIELD_LABELS: Dict[str, str] = {
    "state": "Title State",
    "title_number": "Title Number",
    "vin": "Vehicle Identification Number (VIN)",
    "vehicle_year": "Vehicle Year",
    "make": "Make",
    "model": "Model",
    "body_style": "Body Style",
    "color": "Color",
    "odometer": "Odometer Reading",
    "owner_name": "Owner Name",
    "owner_address": "Owner Address",
    "purchase_price": "Purchase Price ($)",
    "sale_date": "Date of Sale",
    "issue_date": "Title Issue Date",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _output_path(record: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Return a deterministic file path for the PDF."""
    state = _safe_str(record.get("state")) or "XX"
    title_number = _safe_str(record.get("title_number")) or "UNKNOWN"
    vin = _safe_str(record.get("vin")) or "UNKNOWN"
    filename = f"{state}_{title_number}_{vin}.pdf"
    directory = output_dir or _PDF_DIR
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_nmvitis_pdf(
    record: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Generate a NMVITIS submission PDF for a single title record.

    Parameters
    ----------
    record:
        Dict with title fields (keys matching ``_FIELD_LABELS``).
    output_dir:
        Directory to write the PDF.  Defaults to ``/app/data/pdfs``.

    Returns
    -------
    str
        Absolute path of the generated PDF file.
    """
    path = _output_path(record, output_dir)

    doc = SimpleDocTemplate(
        path,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Title / Header ────────────────────────────────────────────────────────
    elements.append(
        Paragraph(
            "NMVITIS Vehicle Title Submission",
            styles["Title"],
        )
    )
    elements.append(Spacer(1, 0.15 * inch))

    record_id = record.get("id")
    if record_id is not None:
        elements.append(
            Paragraph(f"Record ID: {record_id}", styles["Normal"])
        )
        elements.append(Spacer(1, 0.10 * inch))

    # ── Field table ──────────────────────────────────────────────────────────
    table_data = [["Field", "Value"]]
    for field, label in _FIELD_LABELS.items():
        value = _safe_str(record.get(field))
        table_data.append([label, value if value else "—"])

    col_widths = [2.8 * inch, 4.2 * inch]
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                # Data rows – alternating background
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 0.25 * inch))

    # ── Validation status ─────────────────────────────────────────────────────
    is_validated = record.get("is_validated", False)
    status_text = (
        "<font color='green'><b>VALIDATED</b></font>"
        if is_validated
        else "<font color='red'><b>PENDING VALIDATION</b></font>"
    )
    elements.append(Paragraph(f"Status: {status_text}", styles["Normal"]))

    validation_errors = record.get("validation_errors")
    if validation_errors:
        elements.append(Spacer(1, 0.08 * inch))
        if isinstance(validation_errors, list):
            error_text = "; ".join(str(e) for e in validation_errors)
        else:
            error_text = str(validation_errors)
        elements.append(
            Paragraph(
                f"<font color='red'>Errors: {error_text}</font>",
                styles["Normal"],
            )
        )

    doc.build(elements)
    return path


# ---------------------------------------------------------------------------
# MV-7 form generation
# ---------------------------------------------------------------------------

# Mapping from record field names to MV-7 AcroForm field names.
# The MV-7 form contains 30 numbered rows (01-30); each row has:
#   State{nn}               – two-letter state of title
#   "Pennsylvania and/or …{nn}" – title/certificate number
#   "Enter first eight …{nn}"   – first 8 chars of owner last name / business name
#   "Date flattened …{nn}"      – date processed
# Header fields: Text1 (business name), Text2 (MV agent number),
#   "Street Address City State Zip Code" (dealer address).
_MV7_TITLE_NUMBER_FIELD = (
    "Pennsylvania andor OutofState Certificate of TitleSalvage or "
    "Pennsylvania Nonrepairable Certificate Number Do not include letter{nn}"
)
_MV7_OWNER_FIELD = (
    "Enter first eight letters of last name or business name of original title holder{nn}"
)
_MV7_DATE_FIELD = "Date flattened crushed or processed{nn}"
_MV7_STATE_FIELD = "State{nn}"


def _mv7_output_path(record: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Return a deterministic output path for the filled MV-7 PDF."""
    record_id = record.get("id") or "unknown"
    vin = _safe_str(record.get("vin")) or "UNKNOWN"
    filename = f"mv7_{record_id}_{vin}.pdf"
    directory = output_dir or _PDF_DIR
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def generate_mv7_pdf(
    record: Dict[str, Any],
    output_dir: Optional[str] = None,
    row: int = 1,
) -> str:
    """Fill the MV-7 Scrap/Salvage Certificate form for a single record.

    Opens ``BLANK-MV7-FORM.pdf`` (shipped with the application), fills the
    AcroForm fields for *row* (1–30, default 1) with data from *record*, and
    saves the result to *output_dir*.

    Parameters
    ----------
    record:
        Title record dict (from ``get_record_by_id``).
    output_dir:
        Directory to write the PDF.  Defaults to ``/app/data/pdfs``.
    row:
        Which table row (1–30) to populate.  Defaults to 1.

    Returns
    -------
    str
        Absolute path of the generated PDF file.

    Raises
    ------
    FileNotFoundError
        If ``BLANK-MV7-FORM.pdf`` is not found.
    ValueError
        If *row* is outside 1–30.
    """
    if not 1 <= row <= 30:
        raise ValueError(f"row must be between 1 and 30, got {row}")

    if not os.path.exists(_MV7_BLANK):
        raise FileNotFoundError(f"Blank MV-7 form not found at {_MV7_BLANK}")

    doc = fitz.open(_MV7_BLANK)
    page = doc[0]

    # ── Header fields ────────────────────────────────────────────────────────
    plant = _safe_str(record.get("plant_name"))
    dismantler_lic = _safe_str(record.get("dismantler_license"))
    provider = _safe_str(record.get("provider_id"))

    _set_widget(page, "Text1", plant)
    _set_widget(page, "Text2", dismantler_lic)
    _set_widget(page, "Text3", provider)

    # ── Row fields ───────────────────────────────────────────────────────────
    nn = f"{row:02d}"
    state_val = _safe_str(record.get("state"))
    title_number_val = _safe_str(record.get("title_number"))
    owner_val = (_safe_str(record.get("owner_name")) or "")[:8].upper()
    # Prefer sale_date, fall back to issue_date or today.
    date_val = (
        _safe_str(record.get("sale_date"))
        or _safe_str(record.get("issue_date"))
    )

    _set_widget(page, _MV7_STATE_FIELD.format(nn=nn), state_val)
    _set_widget(page, _MV7_TITLE_NUMBER_FIELD.format(nn=nn), title_number_val)
    _set_widget(page, _MV7_OWNER_FIELD.format(nn=nn), owner_val)
    _set_widget(page, _MV7_DATE_FIELD.format(nn=nn), date_val)

    output_path = _mv7_output_path(record, output_dir)
    doc.save(output_path)
    doc.close()
    return output_path


def _set_widget(page: fitz.Page, field_name: str, value: str) -> None:
    """Set the value of a named AcroForm widget on *page*, if it exists."""
    for widget in page.widgets():
        if widget.field_name == field_name:
            widget.field_value = value
            widget.update()
            return
