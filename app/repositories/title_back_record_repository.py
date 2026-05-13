"""Repository for persisting and retrieving back-of-title records."""

from dataclasses import replace
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from app.models.title_back_record import TitleBackRecord


class TitleBackRecordRepository:
    """Encapsulate database operations for :class:`TitleBackRecord`."""

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection

    def save(self, record: TitleBackRecord) -> TitleBackRecord:
        """Insert or update the back-of-title record for a title.

        The ``title_back_records`` table has a UNIQUE constraint on
        ``title_record_id``, so this is an upsert.  Returns the saved record
        with its assigned ``id``.
        """
        created_at = record.created_at or datetime.utcnow().isoformat()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO title_back_records (
                    title_record_id, odometer_at_sale, buyer_name, buyer_address,
                    seller_name, sale_price, sale_date, notes, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (title_record_id) DO UPDATE SET
                    odometer_at_sale = EXCLUDED.odometer_at_sale,
                    buyer_name       = EXCLUDED.buyer_name,
                    buyer_address    = EXCLUDED.buyer_address,
                    seller_name      = EXCLUDED.seller_name,
                    sale_price       = EXCLUDED.sale_price,
                    sale_date        = EXCLUDED.sale_date,
                    notes            = EXCLUDED.notes,
                    created_at       = EXCLUDED.created_at
                RETURNING id
                """,
                (
                    record.title_record_id,
                    record.odometer_at_sale,
                    record.buyer_name,
                    record.buyer_address,
                    record.seller_name,
                    record.sale_price,
                    record.sale_date,
                    record.notes,
                    created_at,
                ),
            )
            back_id = cursor.fetchone()[0]
        self.connection.commit()
        return replace(record, id=back_id, created_at=created_at)

    def get_by_title_record_id(self, title_record_id: int) -> Optional[TitleBackRecord]:
        """Return the back-of-title record for the given front record, or None."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM title_back_records WHERE title_record_id = %s",
                (title_record_id,),
            )
            row = cursor.fetchone()
        return TitleBackRecord.from_dict(dict(row)) if row else None
