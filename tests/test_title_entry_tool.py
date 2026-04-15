import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from title_entry_tool import (
    export_validated_to_csv,
    initialize_database,
    insert_title_record,
    validate_record,
)


def _make_connection(cursor_rows=None, fetchone_return=None):
    """Return a mock psycopg2-style connection/cursor pair.

    cursor_rows     – list of dicts returned by cursor.fetchall()
    fetchone_return – tuple returned by cursor.fetchone()
    """
    cursor = mock.MagicMock()
    cursor.fetchone.return_value = fetchone_return or (1,)
    cursor.fetchall.return_value = cursor_rows or []

    cm = mock.MagicMock()
    cm.__enter__ = mock.Mock(return_value=cursor)
    cm.__exit__ = mock.Mock(return_value=False)

    connection = mock.MagicMock()
    connection.cursor.return_value = cm
    return connection, cursor


class TitleEntryToolTests(unittest.TestCase):
    def test_validate_record_returns_errors_for_invalid_inputs(self) -> None:
        errors = validate_record("12", "1HGCM82633A00435I", 1700)
        self.assertEqual(3, len(errors))

    def test_insert_title_record_marks_validated_record(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(42,))

        result = insert_title_record(
            connection,
            state="NM",
            title_number="abc-1234",
            vin="1HGCM82633A004352",
            vehicle_year=2003,
            ocr_text="OCR SAMPLE",
        )

        self.assertTrue(result["is_validated"])
        self.assertEqual(42, result["id"])
        self.assertEqual([], result["validation_errors"])
        connection.commit.assert_called_once()

        # Verify normalized values were passed to the INSERT
        call_args = cursor.execute.call_args
        params = call_args[0][1]
        self.assertEqual("NM", params[0])
        self.assertEqual("ABC1234", params[1])
        self.assertEqual("1HGCM82633A004352", params[2])
        self.assertEqual(2003, params[3])
        self.assertEqual(1, params[5])  # is_validated

    def test_insert_title_record_marks_invalid_record(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(7,))

        result = insert_title_record(
            connection,
            state="TX",
            title_number="A",
            vin="123",
            vehicle_year=datetime.utcnow().year + 5,
        )

        self.assertFalse(result["is_validated"])
        self.assertGreater(len(result["validation_errors"]), 0)
        self.assertTrue(
            any("VIN must be exactly 17 characters" in e for e in result["validation_errors"])
        )

        call_args = cursor.execute.call_args
        params = call_args[0][1]
        self.assertEqual(0, params[5])  # is_validated
        self.assertIn("VIN must be exactly 17 characters", params[6])  # validation_errors

    def test_export_validated_to_csv_only_exports_valid_records(self) -> None:
        fake_rows = [
            {"state": "NM", "title_number": "NM001", "vin": "1HGCM82633A004352", "vehicle_year": 2003},
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "records.csv"
            exported_count = export_validated_to_csv(connection, str(output_path))

            self.assertEqual(1, exported_count)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(1, len(rows))
        self.assertEqual("NM", rows[0]["state"])
        self.assertEqual("NM001", rows[0]["title_number"])

    def test_initialize_database_executes_create_table(self) -> None:
        connection, cursor = _make_connection()
        initialize_database(connection)
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS title_records", sql)
        connection.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
