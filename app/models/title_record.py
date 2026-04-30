"""TitleRecord data model and field metadata.

Single source of truth for all title field definitions, display labels,
validation constants, and validation logic.  Every other module that needs
to enumerate title fields should import from here rather than re-declaring
its own tuple/dict.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Field classification constants
# ---------------------------------------------------------------------------

#: Core NMVITIS fields detected in every extraction pass.
CORE_FIELDS: Tuple[str, ...] = ("state", "title_number", "vin", "vehicle_year")

#: AI-extracted extended fields (NMVITIS beyond the four core anchors).
EXTENDED_FIELDS: Tuple[str, ...] = (
    "make",
    "model",
    "body_style",
    "color",
    "odometer",
    "owner_name",
    "owner_address",
    "purchase_price",
    "sale_date",
    "issue_date",
)

#: All 14 NMVITIS fields in canonical order (core + extended).
ALL_FIELDS: Tuple[str, ...] = CORE_FIELDS + EXTENDED_FIELDS

#: Operational fields entered manually by staff (never AI-extracted or correction-tracked).
OPERATIONAL_FIELDS: Tuple[str, ...] = (
    "provider_id",
    "state_of_plant",
    "dismantler_license",
    "plant_name",
    "description",
    "condition",
    "stock_number",
    "location_status",
    "purchased_from",
    "sold_to",
)

#: Human-readable display labels for the 14 NMVITIS fields (used by PDF generator).
FIELD_LABELS: Dict[str, str] = {
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
# VIN validation constants
# ---------------------------------------------------------------------------

VIN_ALLOWED: set = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

VIN_TRANSLITERATION: Dict[str, int] = {
    **{str(i): i for i in range(10)},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}

VIN_WEIGHTS: List[int] = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_title_number(title_number: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", title_number.upper())


def validate_title_number(title_number: str) -> Optional[str]:
    """Return an error string, or None when the title number is valid."""
    normalized = _normalize_title_number(title_number)
    if len(normalized) < 3:
        return "Title number must contain at least 3 alphanumeric characters"
    if len(normalized) > 20:
        return "Title number cannot exceed 20 alphanumeric characters"
    return None


def _compute_vin_check_digit(vin: str) -> str:
    total = 0
    for index, character in enumerate(vin):
        total += VIN_TRANSLITERATION[character] * VIN_WEIGHTS[index]
    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def validate_vin(vin: str) -> Optional[str]:
    """Return an error string, or None when the VIN is valid."""
    normalized = vin.strip().upper()
    if len(normalized) != 17:
        return "VIN must be exactly 17 characters"
    if any(character not in VIN_ALLOWED for character in normalized):
        return "VIN contains invalid characters"
    expected_check_digit = _compute_vin_check_digit(normalized)
    if normalized[8] != expected_check_digit:
        return "VIN check digit is invalid"
    return None


def validate_vehicle_year(vehicle_year: int) -> Optional[str]:
    """Return an error string, or None when the vehicle year is valid."""
    current_year = datetime.utcnow().year
    if vehicle_year < 1886:
        return "Vehicle year cannot be earlier than 1886"
    if vehicle_year > current_year + 1:
        return "Vehicle year cannot be more than one year in the future"
    return None


def validate_record(
    title_number: str,
    vin: str,
    vehicle_year: int,
) -> List[str]:
    """Return a list of validation error strings (empty when all fields are valid)."""
    errors: List[str] = []
    for validator, value in (
        (validate_title_number, title_number),
        (validate_vin, vin),
        (validate_vehicle_year, vehicle_year),
    ):
        error = validator(value)
        if error:
            errors.append(error)
    return errors


# ---------------------------------------------------------------------------
# TitleRecord data model
# ---------------------------------------------------------------------------


@dataclass
class TitleRecord:
    """Typed representation of a vehicle title record.

    Serves as the canonical data container for a title across the extraction,
    storage, review, and export pipelines.  Field names mirror the
    ``title_records`` database table columns exactly, allowing ``to_dict()``
    and ``from_dict()`` round-trips without any key translation.
    """

    # ── Core NMVITIS fields ─────────────────────────────────────────────────
    state: str = ""
    title_number: str = ""
    vin: str = ""
    vehicle_year: int = 0

    # ── AI-extracted extended fields ─────────────────────────────────────────
    make: Optional[str] = None
    model: Optional[str] = None
    body_style: Optional[str] = None
    color: Optional[str] = None
    odometer: Optional[int] = None
    owner_name: Optional[str] = None
    owner_address: Optional[str] = None
    purchase_price: Optional[float] = None
    sale_date: Optional[str] = None
    issue_date: Optional[str] = None

    # ── Operational fields (manually entered by staff) ────────────────────────
    provider_id: Optional[str] = None
    state_of_plant: Optional[str] = None
    dismantler_license: Optional[str] = None
    plant_name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    stock_number: Optional[str] = None
    location_status: Optional[str] = None
    purchased_from: Optional[str] = None
    sold_to: Optional[str] = None

    # ── Metadata ──────────────────────────────────────────────────────────────
    # sort_order is a UI display-ordering field managed by the reorder endpoint;
    # it is intentionally excluded from OPERATIONAL_FIELDS (which covers only
    # staff-entered title data) and from ALL_FIELDS (which covers NMVITIS fields).
    sort_order: Optional[int] = None
    state_layout_version: Optional[str] = None
    source_file_path: Optional[str] = None
    ocr_text: str = ""
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    created_at: str = ""
    id: Optional[int] = None

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Run all field validators and return a list of error strings."""
        return validate_record(self.title_number, self.vin, self.vehicle_year)

    # ── Serialisation helpers ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation suitable for DB insertion and JSON."""
        return {
            "id": self.id,
            "state": self.state,
            "title_number": self.title_number,
            "vin": self.vin,
            "vehicle_year": self.vehicle_year,
            "make": self.make,
            "model": self.model,
            "body_style": self.body_style,
            "color": self.color,
            "odometer": self.odometer,
            "owner_name": self.owner_name,
            "owner_address": self.owner_address,
            "purchase_price": self.purchase_price,
            "sale_date": self.sale_date,
            "issue_date": self.issue_date,
            "provider_id": self.provider_id,
            "state_of_plant": self.state_of_plant,
            "dismantler_license": self.dismantler_license,
            "plant_name": self.plant_name,
            "description": self.description,
            "condition": self.condition,
            "stock_number": self.stock_number,
            "location_status": self.location_status,
            "purchased_from": self.purchased_from,
            "sold_to": self.sold_to,
            "sort_order": self.sort_order,
            "state_layout_version": self.state_layout_version,
            "source_file_path": self.source_file_path,
            "ocr_text": self.ocr_text,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TitleRecord":
        """Construct a TitleRecord from a plain dict (e.g. a DB row dict)."""
        errors_raw = d.get("validation_errors")
        if isinstance(errors_raw, str):
            errors: List[str] = [e.strip() for e in errors_raw.split("|") if e.strip()]
        elif isinstance(errors_raw, list):
            errors = list(errors_raw)
        else:
            errors = []
        return cls(
            id=d.get("id"),
            state=d.get("state") or "",
            title_number=d.get("title_number") or "",
            vin=d.get("vin") or "",
            vehicle_year=int(d["vehicle_year"]) if d.get("vehicle_year") is not None else 0,
            make=d.get("make"),
            model=d.get("model"),
            body_style=d.get("body_style"),
            color=d.get("color"),
            odometer=d.get("odometer"),
            owner_name=d.get("owner_name"),
            owner_address=d.get("owner_address"),
            purchase_price=(
                float(d["purchase_price"]) if d.get("purchase_price") is not None else None
            ),
            sale_date=d.get("sale_date"),
            issue_date=d.get("issue_date"),
            provider_id=d.get("provider_id"),
            state_of_plant=d.get("state_of_plant"),
            dismantler_license=d.get("dismantler_license"),
            plant_name=d.get("plant_name"),
            description=d.get("description"),
            condition=d.get("condition"),
            stock_number=d.get("stock_number"),
            location_status=d.get("location_status"),
            purchased_from=d.get("purchased_from"),
            sold_to=d.get("sold_to"),
            sort_order=d.get("sort_order"),
            state_layout_version=d.get("state_layout_version"),
            source_file_path=d.get("source_file_path"),
            ocr_text=d.get("ocr_text") or "",
            is_validated=bool(d.get("is_validated")),
            validation_errors=errors,
            created_at=d.get("created_at") or "",
        )
