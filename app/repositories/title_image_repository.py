"""Repository for persisting and retrieving title image records."""

from dataclasses import replace
from datetime import datetime
from typing import List, Optional

import psycopg2
import psycopg2.extras

from app.models.title_image import TitleImage


class TitleImageRepository:
    """Encapsulate database operations for :class:`TitleImage`.

    Each ``title_images`` row represents one uploaded file (image or PDF) that
    is linked to a title record.  A single title record may have multiple
    associated images (e.g. original PDF plus rendered page previews).
    """

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection

    def create(self, record: TitleImage) -> TitleImage:
        """Insert a new title image record and return it with its assigned id."""
        created_at = record.created_at or datetime.utcnow().isoformat()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO title_images (
                    title_record_id, file_path, file_hash, mime_type,
                    original_filename, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    record.title_record_id,
                    record.file_path,
                    record.file_hash,
                    record.mime_type,
                    record.original_filename,
                    created_at,
                ),
            )
            image_id = cursor.fetchone()[0]
        self.connection.commit()
        return replace(record, id=image_id, created_at=created_at)

    def get_for_record(self, title_record_id: int) -> Optional[TitleImage]:
        """Return the most recently added image for a title record, or None."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT * FROM title_images
                WHERE title_record_id = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (title_record_id,),
            )
            row = cursor.fetchone()
        return TitleImage.from_dict(dict(row)) if row else None

    def list_for_record(self, title_record_id: int) -> List[TitleImage]:
        """Return all images for a title record, newest first."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM title_images WHERE title_record_id = %s ORDER BY id DESC",
                (title_record_id,),
            )
            rows = cursor.fetchall()
        return [TitleImage.from_dict(dict(row)) for row in rows]
