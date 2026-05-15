"""TitleImage data model.

Represents a file upload (image or PDF) associated with a title record,
providing a normalised reference for the upload file path, type, and metadata.
Separating image/document references from the title-domain data in
``title_records`` keeps that table focused on NMVITIS title fields only.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TitleImage:
    """A file upload linked to a title record.

    Field descriptions
    ------------------
    title_record_id : int
        Foreign key referencing ``title_records.id``.
    file_path : str
        Absolute path to the saved upload on the server filesystem.
    file_hash : str, optional
        SHA-256 hex digest of the file content, when computed.
    mime_type : str, optional
        Detected MIME type (e.g. ``"image/png"``, ``"application/pdf"``).
    original_filename : str, optional
        Sanitised original filename provided by the uploader.
    created_at : str
        ISO-8601 timestamp of when the record was created.
    """

    id: Optional[int] = None
    title_record_id: Optional[int] = None
    file_path: str = ""
    file_hash: Optional[str] = None
    mime_type: Optional[str] = None
    original_filename: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation or DB insertion."""
        return {
            "id": self.id,
            "title_record_id": self.title_record_id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "mime_type": self.mime_type,
            "original_filename": self.original_filename,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TitleImage":
        """Construct a TitleImage from a plain dict (e.g. a DB row dict)."""
        return cls(
            id=d.get("id"),
            title_record_id=d.get("title_record_id"),
            file_path=d.get("file_path") or "",
            file_hash=d.get("file_hash"),
            mime_type=d.get("mime_type"),
            original_filename=d.get("original_filename"),
            created_at=d.get("created_at") or "",
        )
