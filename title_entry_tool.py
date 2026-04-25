import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extensions
import psycopg2.extras
import psycopg2.sql

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
VIN_TRANSLITERATION = {
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
VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def create_connection_from_env() -> psycopg2.extensions.connection:
    """Create a Postgres connection using environment variables.

    Environment variables:
        DB_HOST     – hostname of the Postgres server (default: localhost)
        DB_PORT     – port number (default: 5432)
        DB_NAME     – database name (default: titles)
        DB_USER     – database user (default: postgres)
        DB_PASSWORD – database password (default: empty string)
    """
    try:
        port = int(os.getenv("DB_PORT", "5432"))
    except ValueError:
        port = 5432
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=port,
        dbname=os.getenv("DB_NAME", "titles"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def initialize_database(connection: psycopg2.extensions.connection) -> None:
    with connection.cursor() as cursor:
        # Core title records table – extended with NMVITIS fields.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS title_records (
                id SERIAL PRIMARY KEY,
                state TEXT NOT NULL,
                title_number TEXT NOT NULL,
                vin TEXT NOT NULL,
                vehicle_year INTEGER NOT NULL,
                make TEXT,
                model TEXT,
                body_style TEXT,
                color TEXT,
                odometer INTEGER,
                owner_name TEXT,
                owner_address TEXT,
                purchase_price NUMERIC(12,2),
                sale_date TEXT,
                issue_date TEXT,
                state_layout_version TEXT,
                source_file_path TEXT,
                ocr_text TEXT,
                is_validated INTEGER NOT NULL,
                validation_errors TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Idempotent additions for deployments that pre-date this version.
        _extended_columns = [
            ("make", "TEXT"),
            ("model", "TEXT"),
            ("body_style", "TEXT"),
            ("color", "TEXT"),
            ("odometer", "INTEGER"),
            ("owner_name", "TEXT"),
            ("owner_address", "TEXT"),
            ("purchase_price", "NUMERIC(12,2)"),
            ("sale_date", "TEXT"),
            ("issue_date", "TEXT"),
            ("state_layout_version", "TEXT"),
            ("source_file_path", "TEXT"),
        ]
        # Use a fixed allowlist of column definitions to prevent any SQL injection.
        _ALLOWED_EXTENDED_COLUMNS = {
            "make", "model", "body_style", "color", "odometer",
            "owner_name", "owner_address", "purchase_price",
            "sale_date", "issue_date", "state_layout_version", "source_file_path",
        }
        for col_name, col_type in _extended_columns:
            if col_name not in _ALLOWED_EXTENDED_COLUMNS:
                continue  # Skip any unknown column name as a safety guard.
            # Use psycopg2.sql.Identifier to safely compose the identifier
            # even though col_name is already validated above.
            cursor.execute(
                psycopg2.sql.SQL(
                    "ALTER TABLE title_records ADD COLUMN IF NOT EXISTS {} {}"
                ).format(
                    psycopg2.sql.Identifier(col_name),
                    psycopg2.sql.SQL(col_type),
                )
            )

        # Corrections table – ground-truth annotation workflow.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id SERIAL PRIMARY KEY,
                record_id INTEGER REFERENCES title_records(id),
                image_hash TEXT,
                field_name TEXT NOT NULL,
                original_value TEXT,
                corrected_value TEXT,
                is_ground_truth INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )

        # Back-of-title records table.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS title_back_records (
                id SERIAL PRIMARY KEY,
                title_record_id INTEGER REFERENCES title_records(id) UNIQUE,
                odometer_at_sale INTEGER,
                buyer_name TEXT,
                buyer_address TEXT,
                seller_name TEXT,
                sale_price NUMERIC(12,2),
                sale_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
    connection.commit()


def _normalize_title_number(title_number: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", title_number.upper())


def validate_title_number(title_number: str) -> Optional[str]:
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


def insert_title_record(
    connection: psycopg2.extensions.connection,
    state: str,
    title_number: str,
    vin: str,
    vehicle_year: int,
    ocr_text: str = "",
    make: Optional[str] = None,
    model: Optional[str] = None,
    body_style: Optional[str] = None,
    color: Optional[str] = None,
    odometer: Optional[int] = None,
    owner_name: Optional[str] = None,
    owner_address: Optional[str] = None,
    purchase_price: Optional[float] = None,
    sale_date: Optional[str] = None,
    issue_date: Optional[str] = None,
    state_layout_version: Optional[str] = None,
    source_file_path: Optional[str] = None,
) -> Dict[str, object]:
    normalized_title_number = _normalize_title_number(title_number)
    normalized_vin = vin.strip().upper()
    errors = validate_record(normalized_title_number, normalized_vin, vehicle_year)
    validation_errors = " | ".join(errors) if errors else None

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO title_records (
                state, title_number, vin, vehicle_year,
                make, model, body_style, color, odometer,
                owner_name, owner_address, purchase_price,
                sale_date, issue_date, state_layout_version,
                source_file_path, ocr_text,
                is_validated, validation_errors, created_at
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s
            )
            RETURNING id
            """,
            (
                state.strip().upper(),
                normalized_title_number,
                normalized_vin,
                vehicle_year,
                make,
                model,
                body_style,
                color,
                odometer,
                owner_name,
                owner_address,
                purchase_price,
                sale_date,
                issue_date,
                state_layout_version,
                source_file_path,
                ocr_text,
                0 if errors else 1,
                validation_errors,
                datetime.utcnow().isoformat(),
            ),
        )
        row_id = cursor.fetchone()[0]
    connection.commit()
    return {
        "id": row_id,
        "is_validated": not errors,
        "validation_errors": errors,
    }


def export_validated_to_csv(connection: psycopg2.extensions.connection, csv_path: str) -> int:
    """Export all validated title records to a NMVITIS-aligned CSV file."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT
                state, title_number, vin, vehicle_year,
                make, model, body_style, color, odometer,
                owner_name, owner_address, purchase_price,
                sale_date, issue_date
            FROM title_records
            WHERE is_validated = 1
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()

    fieldnames = [
        "state",
        "title_number",
        "vin",
        "vehicle_year",
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
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (dict(row).get(k) or "") for k in fieldnames})

    return len(rows)


def export_validated_to_csv_by_date(
    connection: psycopg2.extensions.connection,
    csv_path: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> int:
    """Export validated records filtered by ``created_at`` date range.

    Parameters
    ----------
    date_from / date_to:
        ISO-8601 date strings (``YYYY-MM-DD``).  Both are optional;
        omitting them falls back to :func:`export_validated_to_csv`.
    """
    conditions = ["is_validated = 1"]
    params: List[Any] = []
    if date_from:
        conditions.append("created_at >= %s")
        params.append(date_from)
    if date_to:
        # Advance to the day after date_to so ``created_at < <next_day>``
        # includes the full last day without fragile string concatenation.
        try:
            from datetime import date as _date, timedelta
            next_day = (
                _date.fromisoformat(date_to) + timedelta(days=1)
            ).isoformat()
        except ValueError:
            next_day = date_to
        conditions.append("created_at < %s")
        params.append(next_day)

    where_clause = " AND ".join(conditions)
    fieldnames = [
        "state", "title_number", "vin", "vehicle_year",
        "make", "model", "body_style", "color", "odometer",
        "owner_name", "owner_address", "purchase_price",
        "sale_date", "issue_date",
    ]
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            f"""
            SELECT
                state, title_number, vin, vehicle_year,
                make, model, body_style, color, odometer,
                owner_name, owner_address, purchase_price,
                sale_date, issue_date
            FROM title_records
            WHERE {where_clause}
            ORDER BY id ASC
            """,
            params or None,
        )
        rows = cursor.fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (dict(row).get(k) or "") for k in fieldnames})

    return len(rows)


# ---------------------------------------------------------------------------
# Record lookup
# ---------------------------------------------------------------------------


def get_record_by_id(
    connection: psycopg2.extensions.connection, record_id: int
) -> Optional[Dict[str, Any]]:
    """Return a title record as a dict, or None if not found."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM title_records WHERE id = %s",
            (record_id,),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def list_records(
    connection: psycopg2.extensions.connection,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return a page of title records ordered by id DESC."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT id, state, title_number, vin, vehicle_year,
                   make, model, color, is_validated, created_at
            FROM title_records
            ORDER BY id DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def insert_correction(
    connection: psycopg2.extensions.connection,
    record_id: int,
    field_name: str,
    original_value: Optional[str],
    corrected_value: Optional[str],
    image_hash: Optional[str] = None,
    is_ground_truth: bool = False,
) -> int:
    """Record a field-level correction for a title record.

    Returns the new correction ``id``.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO corrections (
                record_id, image_hash, field_name,
                original_value, corrected_value, is_ground_truth, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                record_id,
                image_hash,
                field_name,
                original_value,
                corrected_value,
                1 if is_ground_truth else 0,
                datetime.utcnow().isoformat(),
            ),
        )
        correction_id = cursor.fetchone()[0]
    connection.commit()
    return correction_id


def approve_correction(
    connection: psycopg2.extensions.connection, correction_id: int
) -> None:
    """Mark a correction as ground truth."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE corrections SET is_ground_truth = 1 WHERE id = %s",
            (correction_id,),
        )
    connection.commit()


def get_corrections_for_record(
    connection: psycopg2.extensions.connection, record_id: int
) -> List[Dict[str, Any]]:
    """Return all corrections for a given title record."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM corrections WHERE record_id = %s ORDER BY id ASC",
            (record_id,),
        )
        return [dict(r) for r in cursor.fetchall()]


def list_ground_truth_corrections(
    connection: psycopg2.extensions.connection,
) -> List[Dict[str, Any]]:
    """Return all corrections flagged as ground truth (for training export)."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT c.*, r.source_file_path
            FROM corrections c
            JOIN title_records r ON c.record_id = r.id
            WHERE c.is_ground_truth = 1
            ORDER BY c.id ASC
            """
        )
        return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# NMVITIS rejection import
# ---------------------------------------------------------------------------


def import_nmvitis_rejections(
    connection: psycopg2.extensions.connection,
    csv_path: str,
) -> int:
    """Import a NMVITIS rejection CSV and create correction entries for each
    rejected record.

    The CSV is expected to have at minimum a ``vin`` column (used to look up
    the local record) and optional ``error_field`` / ``error_message`` columns.

    Returns the number of corrections created.
    """
    corrections_created = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vin = (row.get("vin") or "").strip().upper()
            if not vin:
                continue
            with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id FROM title_records WHERE vin = %s ORDER BY id DESC LIMIT 1",
                    (vin,),
                )
                record_row = cursor.fetchone()
            if not record_row:
                continue
            record_id = record_row["id"]
            field_name = (row.get("error_field") or "").strip() or "unknown"
            error_message = (row.get("error_message") or "").strip()
            insert_correction(
                connection,
                record_id=record_id,
                field_name=field_name,
                original_value=None,
                corrected_value=None,
                is_ground_truth=False,
            )
            # Also flag the record as needing re-validation.
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE title_records SET is_validated = 0 WHERE id = %s",
                    (record_id,),
                )
            connection.commit()
            corrections_created += 1
    return corrections_created


# ---------------------------------------------------------------------------
# Back-of-title records
# ---------------------------------------------------------------------------


def insert_title_back_record(
    connection: psycopg2.extensions.connection,
    title_record_id: int,
    odometer_at_sale: Optional[int] = None,
    buyer_name: Optional[str] = None,
    buyer_address: Optional[str] = None,
    seller_name: Optional[str] = None,
    sale_price: Optional[float] = None,
    sale_date: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert or replace a back-of-title record.

    Returns the new ``id``.
    """
    with connection.cursor() as cursor:
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
                title_record_id,
                odometer_at_sale,
                buyer_name,
                buyer_address,
                seller_name,
                sale_price,
                sale_date,
                notes,
                datetime.utcnow().isoformat(),
            ),
        )
        back_id = cursor.fetchone()[0]
    connection.commit()
    return back_id


def get_title_back_record(
    connection: psycopg2.extensions.connection, title_record_id: int
) -> Optional[Dict[str, Any]]:
    """Return the back-of-title record for a given title record, or None."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM title_back_records WHERE title_record_id = %s",
            (title_record_id,),
        )
        row = cursor.fetchone()
    return dict(row) if row else None
