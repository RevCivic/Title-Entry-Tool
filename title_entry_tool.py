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

from app.models import TitleRecord
from app.repositories import TitleRecordRepository

# VIN validation constants and all field-level validators are the single source
# of truth in the TitleRecord model; import them here for backward compatibility.
from app.models.title_record import (
    VIN_ALLOWED,
    VIN_TRANSLITERATION,
    VIN_WEIGHTS,
    _normalize_title_number,
    validate_title_number,
    validate_vin,
    validate_vehicle_year,
    validate_record,
)


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
                provider_id TEXT,
                state_of_plant TEXT,
                dismantler_license TEXT,
                plant_name TEXT,
                description TEXT,
                condition TEXT,
                stock_number TEXT,
                location_status TEXT,
                purchased_from TEXT,
                sold_to TEXT,
                sort_order INTEGER,
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
            ("provider_id", "TEXT"),
            ("state_of_plant", "TEXT"),
            ("dismantler_license", "TEXT"),
            ("plant_name", "TEXT"),
            ("description", "TEXT"),
            ("condition", "TEXT"),
            ("stock_number", "TEXT"),
            ("location_status", "TEXT"),
            ("purchased_from", "TEXT"),
            ("sold_to", "TEXT"),
            ("sort_order", "INTEGER"),
            ("state_layout_version", "TEXT"),
            ("source_file_path", "TEXT"),
        ]
        # Use a fixed allowlist of column definitions to prevent any SQL injection.
        _ALLOWED_EXTENDED_COLUMNS = {
            "make", "model", "body_style", "color", "odometer",
            "owner_name", "owner_address", "purchase_price",
            "sale_date", "issue_date",
            "provider_id", "state_of_plant", "dismantler_license", "plant_name",
            "description", "condition", "stock_number", "location_status",
            "purchased_from", "sold_to", "sort_order",
            "state_layout_version", "source_file_path",
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

        # Training runs table – records each ground-truth export event.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                exported_at TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                export_path TEXT
            )
            """
        )

        # App settings – generic key/value store for application configuration.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Model definitions – tracks custom models built within the app.
        # Model weights are stored in the ollama_data volume; this table records
        # the metadata so models can be identified and rebuilt across redeployments.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_definitions (
                id SERIAL PRIMARY KEY,
                model_name TEXT UNIQUE NOT NULL,
                base_model TEXT NOT NULL,
                description TEXT,
                corrections_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
    connection.commit()


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
    provider_id: Optional[str] = None,
    state_of_plant: Optional[str] = None,
    dismantler_license: Optional[str] = None,
    plant_name: Optional[str] = None,
    description: Optional[str] = None,
    condition: Optional[str] = None,
    stock_number: Optional[str] = None,
    location_status: Optional[str] = None,
    purchased_from: Optional[str] = None,
    sold_to: Optional[str] = None,
    sort_order: Optional[int] = None,
    state_layout_version: Optional[str] = None,
    source_file_path: Optional[str] = None,
) -> Dict[str, object]:
    repository = TitleRecordRepository(connection)
    record = repository.create(
        TitleRecord(
            state=state,
            title_number=title_number,
            vin=vin,
            vehicle_year=vehicle_year,
            make=make,
            model=model,
            body_style=body_style,
            color=color,
            odometer=odometer,
            owner_name=owner_name,
            owner_address=owner_address,
            purchase_price=purchase_price,
            sale_date=sale_date,
            issue_date=issue_date,
            provider_id=provider_id,
            state_of_plant=state_of_plant,
            dismantler_license=dismantler_license,
            plant_name=plant_name,
            description=description,
            condition=condition,
            stock_number=stock_number,
            location_status=location_status,
            purchased_from=purchased_from,
            sold_to=sold_to,
            sort_order=sort_order,
            state_layout_version=state_layout_version,
            source_file_path=source_file_path,
            ocr_text=ocr_text,
        )
    )
    return {
        "id": record.id,
        "is_validated": record.is_validated,
        "validation_errors": record.validation_errors,
    }


def export_validated_to_csv(connection: psycopg2.extensions.connection, csv_path: str) -> int:
    """Export all validated title records to a NMVITIS-aligned CSV file."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT
                provider_id, vin, title_number, state, state_of_plant,
                dismantler_license, plant_name, make, model, vehicle_year,
                odometer, description, condition, stock_number, location_status,
                purchased_from, created_at, sold_to
            FROM title_records
            WHERE is_validated = 1
            ORDER BY COALESCE(sort_order, id) ASC
            """
        )
        rows = cursor.fetchall()

    fieldnames = [
        "provider_id", "vin", "title_number", "state", "state_of_plant",
        "dismantler_license", "plant_name", "make", "model", "vehicle_year",
        "odometer", "description", "condition", "stock_number", "location_status",
        "purchased_from", "created_at", "sold_to",
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
        "provider_id", "vin", "title_number", "state", "state_of_plant",
        "dismantler_license", "plant_name", "make", "model", "vehicle_year",
        "odometer", "description", "condition", "stock_number", "location_status",
        "purchased_from", "created_at", "sold_to",
    ]
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            f"""
            SELECT
                provider_id, vin, title_number, state, state_of_plant,
                dismantler_license, plant_name, make, model, vehicle_year,
                odometer, description, condition, stock_number, location_status,
                purchased_from, created_at, sold_to
            FROM title_records
            WHERE {where_clause}
            ORDER BY COALESCE(sort_order, id) ASC
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
    record = TitleRecordRepository(connection).get_by_id(record_id)
    return record.to_dict() if record else None


def list_records(
    connection: psycopg2.extensions.connection,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return a page of title records ordered by id DESC (newest first)."""
    return [
        record.to_dict()
        for record in TitleRecordRepository(connection).list(limit=limit, offset=offset)
    ]


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


def update_title_record_fields(
    connection: psycopg2.extensions.connection,
    record_id: int,
    fields: Dict[str, Any],
) -> None:
    """Overwrite specific columns of a title record and re-validate it."""
    TitleRecordRepository(connection).update_fields(record_id, fields)


def update_record_sort_orders(
    connection: psycopg2.extensions.connection,
    orders: List[Dict[str, int]],
) -> None:
    """Bulk-update ``sort_order`` for a list of records.

    Parameters
    ----------
    orders:
        List of dicts, each containing ``id`` (int) and ``sort_order`` (int).
    """
    with connection.cursor() as cursor:
        for item in orders:
            record_id = int(item["id"])
            sort_order = int(item["sort_order"])
            cursor.execute(
                "UPDATE title_records SET sort_order = %s WHERE id = %s",
                (sort_order, record_id),
            )
    connection.commit()


def list_records_for_reorder(
    connection: psycopg2.extensions.connection,
) -> List[Dict[str, Any]]:
    """Return all validated records in current sort order for the reorder UI."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT id, state, title_number, vin, vehicle_year, make, model,
                   stock_number, sort_order
            FROM title_records
            WHERE is_validated = 1
            ORDER BY COALESCE(sort_order, id) ASC
            """
        )
        return [dict(r) for r in cursor.fetchall()]


def get_annotation_queue(
    connection: psycopg2.extensions.connection,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return records needing annotation ordered by priority.

    Priority order:
      1. Fewest ground-truth corrections (unannotated first).
      2. Unvalidated records before validated.
      3. Newest records first within each tier.
    """
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            """
            SELECT
                r.id, r.state, r.title_number, r.vin, r.vehicle_year,
                r.make, r.model, r.color, r.is_validated, r.created_at,
                COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) AS gt_count
            FROM title_records r
            LEFT JOIN corrections c ON c.record_id = r.id
            GROUP BY r.id
            ORDER BY
                COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) ASC,
                r.is_validated ASC,
                r.id DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return [dict(r) for r in cursor.fetchall()]


def get_training_stats(
    connection: psycopg2.extensions.connection,
) -> Dict[str, Any]:
    """Return statistics about the ground-truth training dataset."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            "SELECT COUNT(*) AS total FROM corrections WHERE is_ground_truth = 1"
        )
        row = cursor.fetchone()
        total = int(row["total"]) if row else 0

        cursor.execute(
            """
            SELECT field_name, COUNT(*) AS count
            FROM corrections
            WHERE is_ground_truth = 1
            GROUP BY field_name
            ORDER BY count DESC
            """
        )
        by_field = [dict(r) for r in cursor.fetchall()]

        cursor.execute(
            """
            SELECT r.state, COUNT(c.id) AS count
            FROM corrections c
            JOIN title_records r ON c.record_id = r.id
            WHERE c.is_ground_truth = 1
            GROUP BY r.state
            ORDER BY count DESC
            """
        )
        by_state = [dict(r) for r in cursor.fetchall()]

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_records,
                COUNT(CASE WHEN gt_count >= 1 THEN 1 END) AS ge1,
                COUNT(CASE WHEN gt_count >= 5 THEN 1 END) AS ge5,
                COUNT(CASE WHEN gt_count >= 10 THEN 1 END) AS ge10
            FROM (
                SELECT r.id,
                    COUNT(c.id) FILTER (WHERE c.is_ground_truth = 1) AS gt_count
                FROM title_records r
                LEFT JOIN corrections c ON c.record_id = r.id
                GROUP BY r.id
            ) sub
            """
        )
        cov_row = cursor.fetchone()
        coverage = (
            {k: int(v) for k, v in dict(cov_row).items()}
            if cov_row
            else {"total_records": 0, "ge1": 0, "ge5": 0, "ge10": 0}
        )

    return {
        "total_gt_corrections": total,
        "by_field": by_field,
        "by_state": by_state,
        "coverage": coverage,
    }


def insert_training_run(
    connection: psycopg2.extensions.connection,
    sample_count: int,
    notes: str = "",
    export_path: Optional[str] = None,
) -> int:
    """Record a training-data export event and return its id."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO training_runs (exported_at, sample_count, notes, export_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (datetime.utcnow().isoformat(), sample_count, notes or None, export_path),
        )
        run_id = cursor.fetchone()[0]
    connection.commit()
    return run_id


def list_training_runs(
    connection: psycopg2.extensions.connection,
) -> List[Dict[str, Any]]:
    """Return all training runs, newest first."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM training_runs ORDER BY id DESC")
        return [dict(r) for r in cursor.fetchall()]


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------


def get_app_setting(
    connection: psycopg2.extensions.connection,
    key: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the value of an application setting, or *default* when not set."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
        row = cursor.fetchone()
    return str(row[0]) if row and row[0] is not None else default


def set_app_setting(
    connection: psycopg2.extensions.connection,
    key: str,
    value: str,
) -> None:
    """Upsert an application setting."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
            """,
            (key, value, datetime.utcnow().isoformat()),
        )
    connection.commit()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def upsert_model_definition(
    connection: psycopg2.extensions.connection,
    model_name: str,
    base_model: str,
    description: Optional[str] = None,
    corrections_used: int = 0,
) -> None:
    """Insert or update a custom model definition record."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_definitions (model_name, base_model, description, corrections_used, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (model_name) DO UPDATE
                SET base_model = EXCLUDED.base_model,
                    description = EXCLUDED.description,
                    corrections_used = EXCLUDED.corrections_used
            """,
            (model_name, base_model, description, corrections_used, datetime.utcnow().isoformat()),
        )
    connection.commit()


def list_model_definitions(
    connection: psycopg2.extensions.connection,
) -> List[Dict[str, Any]]:
    """Return all custom model definitions, newest first."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM model_definitions ORDER BY id DESC")
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
# CSV bulk import helpers
# ---------------------------------------------------------------------------


def find_duplicate_record(
    connection: psycopg2.extensions.connection,
    state: str,
    title_number: str,
    vin: str,
) -> Optional[int]:
    """Return the ``id`` of an existing record matching *state* + *title_number* + *vin*.

    Returns ``None`` when no duplicate exists.  The comparison is
    case-insensitive (all values are upper-cased before querying).
    """
    return TitleRecordRepository(connection).find_duplicate(state, title_number, vin)


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
