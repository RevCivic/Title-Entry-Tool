"""Repository for persisting and retrieving field-level corrections."""

from datetime import datetime
from typing import List, Optional

import psycopg2
import psycopg2.extras

from app.models.correction import Correction


class CorrectionRepository:
    """Encapsulate database operations for :class:`Correction`."""

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection

    def create(
        self,
        record_id: int,
        field_name: str,
        original_value: Optional[str],
        corrected_value: Optional[str],
        image_hash: Optional[str] = None,
        image_id: Optional[int] = None,
        is_ground_truth: bool = False,
    ) -> Correction:
        """Insert a new correction and return it with its assigned id."""
        created_at = datetime.utcnow().isoformat()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO corrections (
                    record_id, image_hash, image_id, field_name,
                    original_value, corrected_value, is_ground_truth, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    record_id,
                    image_hash,
                    image_id,
                    field_name,
                    original_value,
                    corrected_value,
                    1 if is_ground_truth else 0,
                    created_at,
                ),
            )
            correction_id = cursor.fetchone()[0]
        self.connection.commit()
        return Correction(
            id=correction_id,
            record_id=record_id,
            image_hash=image_hash,
            image_id=image_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            is_ground_truth=is_ground_truth,
            created_at=created_at,
        )

    def approve(self, correction_id: int) -> None:
        """Mark a correction as ground truth."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE corrections SET is_ground_truth = 1 WHERE id = %s",
                (correction_id,),
            )
        self.connection.commit()

    def list_for_record(self, record_id: int) -> List[Correction]:
        """Return all corrections for a given title record, oldest first."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM corrections WHERE record_id = %s ORDER BY id ASC",
                (record_id,),
            )
            rows = cursor.fetchall()
        return [Correction.from_dict(dict(row)) for row in rows]

    def list_ground_truth(self) -> List[Correction]:
        """Return all ground-truth corrections, including each record's source path.

        The file path is resolved through the ``title_images`` reference table
        when an ``image_id`` FK is set.  For records that pre-date the image
        reference table (i.e. where ``image_id`` is NULL), it falls back to
        ``title_records.source_file_path`` so that existing training exports
        continue to work without a full data migration.
        """
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT c.*,
                    COALESCE(ti.file_path, r.source_file_path) AS source_file_path
                FROM corrections c
                LEFT JOIN title_images ti ON c.image_id = ti.id
                LEFT JOIN title_records r ON c.record_id = r.id
                WHERE c.is_ground_truth = 1
                ORDER BY c.id ASC
                """
            )
            rows = cursor.fetchall()
        return [Correction.from_dict(dict(row)) for row in rows]
