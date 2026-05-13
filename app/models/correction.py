"""Correction data model.

Represents a single field-level correction for a title record,
used to build the ground-truth training dataset.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Correction:
    """A single field-level correction applied to a title record."""

    id: Optional[int] = None
    record_id: Optional[int] = None
    image_hash: Optional[str] = None
    field_name: str = ""
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    is_ground_truth: bool = False
    created_at: str = ""
    # Populated by joins when fetching ground-truth corrections for export/training.
    source_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.record_id,
            "image_hash": self.image_hash,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "is_ground_truth": self.is_ground_truth,
            "created_at": self.created_at,
            "source_file_path": self.source_file_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Correction":
        return cls(
            id=d.get("id"),
            record_id=d.get("record_id"),
            image_hash=d.get("image_hash"),
            field_name=d.get("field_name") or "",
            original_value=d.get("original_value"),
            corrected_value=d.get("corrected_value"),
            is_ground_truth=bool(d.get("is_ground_truth")),
            created_at=d.get("created_at") or "",
            source_file_path=d.get("source_file_path"),
        )
