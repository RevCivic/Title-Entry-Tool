import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from title_entry_tool import (
    create_connection,
    export_validated_to_csv,
    initialize_database,
    insert_title_record,
    validate_record,
)


class TitleEntryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = create_connection(":memory:")
        initialize_database(self.connection)

    def tearDown(self) -> None:
        self.connection.close()

    def test_validate_record_returns_errors_for_invalid_inputs(self) -> None:
        errors = validate_record("12", "1HGCM82633A00435I", 1700)
        self.assertEqual(3, len(errors))

    def test_insert_title_record_marks_validated_record(self) -> None:
        result = insert_title_record(
            self.connection,
            state="NM",
            title_number="abc-1234",
            vin="1HGCM82633A004352",
            vehicle_year=2003,
            ocr_text="OCR SAMPLE",
        )

        self.assertTrue(result["is_validated"])
        row = self.connection.execute(
            "SELECT state, title_number, vin, vehicle_year, is_validated FROM title_records"
        ).fetchone()
        self.assertEqual("NM", row["state"])
        self.assertEqual("ABC1234", row["title_number"])
        self.assertEqual("1HGCM82633A004352", row["vin"])
        self.assertEqual(2003, row["vehicle_year"])
        self.assertEqual(1, row["is_validated"])

    def test_insert_title_record_marks_invalid_record(self) -> None:
        result = insert_title_record(
            self.connection,
            state="TX",
            title_number="A",
            vin="123",
            vehicle_year=datetime.utcnow().year + 5,
        )

        self.assertFalse(result["is_validated"])
        row = self.connection.execute(
            "SELECT is_validated, validation_errors FROM title_records"
        ).fetchone()
        self.assertEqual(0, row["is_validated"])
        self.assertIn("VIN must be exactly 17 characters", row["validation_errors"])

    def test_export_validated_to_csv_only_exports_valid_records(self) -> None:
        insert_title_record(
            self.connection,
            state="NM",
            title_number="NM-001",
            vin="1HGCM82633A004352",
            vehicle_year=2003,
        )
        insert_title_record(
            self.connection,
            state="NM",
            title_number="X",
            vin="BADVIN",
            vehicle_year=3000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "records.csv"
            exported_count = export_validated_to_csv(self.connection, str(output_path))

            self.assertEqual(1, exported_count)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(1, len(rows))
        self.assertEqual("NM", rows[0]["state"])
        self.assertEqual("NM001", rows[0]["title_number"])


if __name__ == "__main__":
    unittest.main()
