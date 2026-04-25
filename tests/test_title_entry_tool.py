import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from title_entry_tool import (
    export_validated_to_csv,
    get_record_by_id,
    initialize_database,
    insert_correction,
    insert_title_back_record,
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
        # is_validated is now at index 17 (after 14 extended fields + state_layout_version + source_file_path + ocr_text)
        self.assertEqual(1, params[17])  # is_validated

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
        self.assertEqual(0, params[17])  # is_validated
        self.assertIn("VIN must be exactly 17 characters", params[18])  # validation_errors

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

    def test_insert_title_record_stores_extended_fields(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(99,))

        result = insert_title_record(
            connection,
            state="NM",
            title_number="TEST1234",
            vin="1HGCM82633A004352",
            vehicle_year=2003,
            make="HONDA",
            model="ACCORD",
            body_style="SDN",
            color="SILVER",
            odometer=85000,
            owner_name="JANE DOE",
            owner_address="123 MAIN ST",
            purchase_price=12500.00,
            sale_date="2024-01-15",
            issue_date="2024-01-20",
        )

        self.assertTrue(result["is_validated"])
        call_args = cursor.execute.call_args
        params = call_args[0][1]
        # Extended fields start at index 4 (after state, title_number, vin, vehicle_year)
        self.assertEqual("HONDA", params[4])   # make
        self.assertEqual("ACCORD", params[5])  # model
        self.assertEqual("SDN", params[6])     # body_style
        self.assertEqual("SILVER", params[7])  # color
        self.assertEqual(85000, params[8])     # odometer

    def test_insert_correction_calls_execute(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(1,))
        cid = insert_correction(
            connection,
            record_id=42,
            field_name="vin",
            original_value="BAD",
            corrected_value="1HGCM82633A004352",
            is_ground_truth=True,
        )
        self.assertEqual(1, cid)
        connection.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO corrections", sql)

    def test_insert_title_back_record_calls_execute(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(5,))
        bid = insert_title_back_record(
            connection,
            title_record_id=42,
            odometer_at_sale=95000,
            buyer_name="BUYER CORP",
            sale_date="2024-06-01",
        )
        self.assertEqual(5, bid)
        connection.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO title_back_records", sql)

    def test_initialize_database_creates_all_tables(self) -> None:
        connection, cursor = _make_connection()
        initialize_database(connection)
        # Should have called execute multiple times (one CREATE TABLE + ALTER + corrections + back)
        self.assertGreater(cursor.execute.call_count, 1)
        # Collect all SQL strings (some may be psycopg2.sql.Composed objects).
        sql_parts = []
        for call in cursor.execute.call_args_list:
            arg = call[0][0]
            sql_parts.append(str(arg) if isinstance(arg, str) else repr(arg))
        all_sql = " ".join(sql_parts)
        self.assertIn("title_records", all_sql)
        self.assertIn("corrections", all_sql)
        self.assertIn("title_back_records", all_sql)


if __name__ == "__main__":
    unittest.main()
