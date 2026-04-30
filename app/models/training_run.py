"""TrainingRun data model.

Records each ground-truth export and LLM fine-tuning event,
providing an audit trail for the model training pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingRun:
    """A single ground-truth export / LLM training event."""

    id: Optional[int] = None
    exported_at: str = ""
    sample_count: int = 0
    notes: Optional[str] = None
    export_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "exported_at": self.exported_at,
            "sample_count": self.sample_count,
            "notes": self.notes,
            "export_path": self.export_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingRun":
        return cls(
            id=d.get("id"),
            exported_at=d.get("exported_at") or "",
            sample_count=int(d.get("sample_count") or 0),
            notes=d.get("notes"),
            export_path=d.get("export_path"),
        )
