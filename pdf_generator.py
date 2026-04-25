"""NMVITIS PDF submission form generator.

Produces a single-page PDF for each title record that can be submitted
to NMVITIS for record-keeping.  Files are named
``{state}_{title_number}_{vin}.pdf`` and saved to the configured data
directory.
"""

import os
from typing import Any, Dict, Optional

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
