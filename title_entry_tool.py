import csv
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

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


def create_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS title_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            title_number TEXT NOT NULL,
            vin TEXT NOT NULL,
            vehicle_year INTEGER NOT NULL,
            ocr_text TEXT,
            is_validated INTEGER NOT NULL,
            validation_errors TEXT,
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
    connection: sqlite3.Connection,
    state: str,
    title_number: str,
    vin: str,
    vehicle_year: int,
    ocr_text: str = "",
) -> Dict[str, object]:
    normalized_title_number = _normalize_title_number(title_number)
    normalized_vin = vin.strip().upper()
    errors = validate_record(normalized_title_number, normalized_vin, vehicle_year)
    validation_errors = " | ".join(errors) if errors else None

    cursor = connection.execute(
        """
        INSERT INTO title_records (
            state, title_number, vin, vehicle_year, ocr_text,
            is_validated, validation_errors, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            state.strip().upper(),
            normalized_title_number,
            normalized_vin,
            vehicle_year,
            ocr_text,
            0 if errors else 1,
            validation_errors,
            datetime.utcnow().isoformat(),
        ),
    )
    connection.commit()
    return {
        "id": cursor.lastrowid,
        "is_validated": not errors,
        "validation_errors": errors,
    }


def export_validated_to_csv(connection: sqlite3.Connection, csv_path: str) -> int:
    rows = connection.execute(
        """
        SELECT state, title_number, vin, vehicle_year
        FROM title_records
        WHERE is_validated = 1
        ORDER BY id ASC
        """
    ).fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["state", "title_number", "vin", "vehicle_year"],
        )
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)

    return len(rows)
