"""Repository for persisting and retrieving title records."""

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.sql

from app.models.title_record import ALL_FIELDS, OPERATIONAL_FIELDS, TitleRecord, _normalize_title_number


class TitleRecordRepository:
    """Encapsulate database operations for :class:`TitleRecord`."""

    UPDATABLE_FIELDS = frozenset(set(ALL_FIELDS) | set(OPERATIONAL_FIELDS))

    def __init__(self, connection: psycopg2.extensions.connection) -> None:
        self.connection = connection

    def create(self, record: TitleRecord) -> TitleRecord:
        normalized = self._normalize(record)
        validation_errors = normalized.validate()
        created_at = normalized.created_at or datetime.utcnow().isoformat()

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO title_records (
                    state, title_number, vin, vehicle_year,
                    make, model, body_style, color, odometer,
                    owner_name, owner_address, purchase_price,
                    sale_date, issue_date,
                    provider_id, state_of_plant, dismantler_license, plant_name,
                    description, condition, stock_number, location_status,
                    purchased_from, sold_to, sort_order,
                    state_layout_version, source_file_path, ocr_text,
                    is_validated, validation_errors, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                RETURNING id
                """,
                (
                    normalized.state,
                    normalized.title_number,
                    normalized.vin,
                    normalized.vehicle_year,
                    normalized.make,
                    normalized.model,
                    normalized.body_style,
                    normalized.color,
                    normalized.odometer,
                    normalized.owner_name,
                    normalized.owner_address,
                    normalized.purchase_price,
                    normalized.sale_date,
                    normalized.issue_date,
                    normalized.provider_id,
                    normalized.state_of_plant,
                    normalized.dismantler_license,
                    normalized.plant_name,
                    normalized.description,
                    normalized.condition,
                    normalized.stock_number,
                    normalized.location_status,
                    normalized.purchased_from,
                    normalized.sold_to,
                    normalized.sort_order,
                    normalized.state_layout_version,
                    normalized.source_file_path,
                    normalized.ocr_text,
                    0 if validation_errors else 1,
                    " | ".join(validation_errors) if validation_errors else None,
                    created_at,
                ),
            )
            record_id = cursor.fetchone()[0]
        self.connection.commit()
        return replace(
            normalized,
            id=record_id,
            is_validated=not validation_errors,
            validation_errors=validation_errors,
            created_at=created_at,
        )

    def get_by_id(self, record_id: int) -> Optional[TitleRecord]:
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM title_records WHERE id = %s", (record_id,))
            row = cursor.fetchone()
        return TitleRecord.from_dict(dict(row)) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> List[TitleRecord]:
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, state, title_number, vin, vehicle_year,
                       make, model, color, is_validated, created_at,
                       stock_number, location_status, sort_order
                FROM title_records
                ORDER BY id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [TitleRecord.from_dict(dict(row)) for row in rows]

    def update_fields(self, record_id: int, fields: Dict[str, Any]) -> None:
        safe_fields = {key: value for key, value in fields.items() if key in self.UPDATABLE_FIELDS}
        if not safe_fields:
            return

        set_clauses = [
            psycopg2.sql.SQL("{} = %s").format(psycopg2.sql.Identifier(key))
            for key in safe_fields
        ]
        query = psycopg2.sql.SQL(
            "UPDATE title_records SET {} WHERE id = %s"
        ).format(psycopg2.sql.SQL(", ").join(set_clauses))

        with self.connection.cursor() as cursor:
            cursor.execute(query, list(safe_fields.values()) + [record_id])

        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT title_number, vin, vehicle_year FROM title_records WHERE id = %s",
                (record_id,),
            )
            row = cursor.fetchone()
        if not row:
            self.connection.commit()
            return
        title_number, vin, vehicle_year = row
        validation_errors = TitleRecord(
            title_number=title_number or "",
            vin=vin or "",
            vehicle_year=vehicle_year or 0,
        ).validate()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE title_records
                SET is_validated = %s, validation_errors = %s
                WHERE id = %s
                """,
                (
                    0 if validation_errors else 1,
                    " | ".join(validation_errors) if validation_errors else None,
                    record_id,
                ),
            )
        self.connection.commit()

    def find_duplicate(self, state: str, title_number: str, vin: str) -> Optional[int]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM title_records
                WHERE state = %s AND title_number = %s AND vin = %s
                LIMIT 1
                """,
                (
                    state.strip().upper(),
                    _normalize_title_number(title_number),
                    vin.strip().upper(),
                ),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else None

    @staticmethod
    def _normalize(record: TitleRecord) -> TitleRecord:
        state = (record.state or "").strip().upper()
        title_number = _normalize_title_number(record.title_number or "")
        vin = (record.vin or "").strip().upper()
        return replace(
            record,
            state=state,
            title_number=title_number,
            vin=vin,
        )
