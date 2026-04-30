"""Data models for the Title Entry Tool.

Public re-exports so callers can use ``from app.models import TitleRecord``
instead of the full submodule path.
"""

from app.models.correction import Correction
from app.models.title_back_record import TitleBackRecord
from app.models.title_record import (
    ALL_FIELDS,
    CORE_FIELDS,
    EXTENDED_FIELDS,
    FIELD_LABELS,
    OPERATIONAL_FIELDS,
    VIN_ALLOWED,
    VIN_TRANSLITERATION,
    VIN_WEIGHTS,
    TitleRecord,
    validate_record,
    validate_title_number,
    validate_vehicle_year,
    validate_vin,
)
from app.models.training_run import TrainingRun

__all__ = [
    "TitleRecord",
    "Correction",
    "TitleBackRecord",
    "TrainingRun",
    "CORE_FIELDS",
    "EXTENDED_FIELDS",
    "ALL_FIELDS",
    "OPERATIONAL_FIELDS",
    "FIELD_LABELS",
    "VIN_ALLOWED",
    "VIN_TRANSLITERATION",
    "VIN_WEIGHTS",
    "validate_title_number",
    "validate_vin",
    "validate_vehicle_year",
    "validate_record",
]
