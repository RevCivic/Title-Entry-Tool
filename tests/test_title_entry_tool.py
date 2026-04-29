import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from title_entry_tool import (
    export_validated_to_csv,
    get_annotation_queue,
    get_app_setting,
    get_record_by_id,
    get_training_stats,
    initialize_database,
    insert_correction,
    insert_title_back_record,
    insert_title_record,
    insert_training_run,
    list_model_definitions,
    list_records_for_reorder,
    list_training_runs,
    set_app_setting,
    update_record_sort_orders,
    update_title_record_fields,
    upsert_model_definition,
    validate_record,
)


def _make_connection(cursor_rows=None, fetchone_return=None, fetchone_explicit_none=False):
    """Return a mock psycopg2-style connection/cursor pair.

    cursor_rows              – list of dicts returned by cursor.fetchall()
    fetchone_return          – tuple returned by cursor.fetchone()
    fetchone_explicit_none   – if True, configure fetchone() to return None
                               (overrides the default (1,) fallback)
    """
    cursor = mock.MagicMock()
    if fetchone_explicit_none:
        cursor.fetchone.return_value = None
    else:
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
        # is_validated is at index 28 (14 core + 11 new operational + state_layout_version
        # + source_file_path + ocr_text = 28)
        self.assertEqual(1, params[28])  # is_validated

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
        self.assertEqual(0, params[28])  # is_validated
        self.assertIn("VIN must be exactly 17 characters", params[29])  # validation_errors

    def test_export_validated_to_csv_only_exports_valid_records(self) -> None:
        fake_rows = [
            {
                "provider_id": "P001", "vin": "1HGCM82633A004352",
                "title_number": "NM001", "state": "NM", "state_of_plant": "NM",
                "dismantler_license": "DL123", "plant_name": "Test Yard",
                "make": "HONDA", "model": "ACCORD", "vehicle_year": 2003,
                "odometer": 85000, "description": None, "condition": "Salvage",
                "stock_number": "S001", "location_status": "In Yard",
                "purchased_from": "Auction", "created_at": "2024-01-01T00:00:00",
                "sold_to": None,
            },
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
        self.assertEqual("P001", rows[0]["provider_id"])
        self.assertEqual("S001", rows[0]["stock_number"])

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

    def test_insert_title_record_stores_operational_fields(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(77,))

        result = insert_title_record(
            connection,
            state="NM",
            title_number="OP1234",
            vin="1HGCM82633A004352",
            vehicle_year=2003,
            provider_id="PROV01",
            state_of_plant="NM",
            dismantler_license="DL9999",
            plant_name="Test Yard",
            description="Blue sedan, front damage",
            condition="Salvage",
            stock_number="S-100",
            location_status="In Yard",
            purchased_from="ABC Auction",
            sold_to="Recycler LLC",
        )

        self.assertTrue(result["is_validated"])
        call_args = cursor.execute.call_args
        params = call_args[0][1]
        # Operational fields start at index 14 (after 14 NMVITIS fields)
        self.assertEqual("PROV01", params[14])       # provider_id
        self.assertEqual("NM", params[15])            # state_of_plant
        self.assertEqual("DL9999", params[16])        # dismantler_license
        self.assertEqual("Test Yard", params[17])     # plant_name
        self.assertEqual("Blue sedan, front damage", params[18])  # description
        self.assertEqual("Salvage", params[19])       # condition
        self.assertEqual("S-100", params[20])         # stock_number
        self.assertEqual("In Yard", params[21])       # location_status
        self.assertEqual("ABC Auction", params[22])   # purchased_from
        self.assertEqual("Recycler LLC", params[23])  # sold_to

    def test_update_record_sort_orders_executes_updates(self) -> None:
        connection, cursor = _make_connection()
        orders = [{"id": 3, "sort_order": 1}, {"id": 1, "sort_order": 2}]
        update_record_sort_orders(connection, orders)
        connection.commit.assert_called_once()
        # Two UPDATE statements should have been called
        execute_calls = cursor.execute.call_args_list
        self.assertEqual(2, len(execute_calls))
        for call in execute_calls:
            sql = call[0][0]
            self.assertIn("sort_order", sql)

    def test_list_records_for_reorder_returns_rows(self) -> None:
        fake_rows = [
            {"id": 1, "state": "NM", "title_number": "T1", "vin": "1HGCM82633A004352",
             "vehicle_year": 2003, "make": "HONDA", "model": "ACCORD",
             "stock_number": "S001", "sort_order": 1},
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        records = list_records_for_reorder(connection)
        self.assertEqual(1, len(records))
        self.assertEqual("S001", records[0]["stock_number"])

    def test_initialize_database_includes_new_operational_columns(self) -> None:
        connection, cursor = _make_connection()
        initialize_database(connection)
        sql_parts = []
        for call in cursor.execute.call_args_list:
            arg = call[0][0]
            sql_parts.append(str(arg) if isinstance(arg, str) else repr(arg))
        all_sql = " ".join(sql_parts)
        # Verify new operational columns are part of the schema setup.
        for col in ("provider_id", "state_of_plant", "dismantler_license",
                    "plant_name", "stock_number", "sort_order"):
            self.assertIn(col, all_sql, msg=f"Column '{col}' not found in DB init SQL")

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
        self.assertIn("training_runs", all_sql)
        self.assertIn("app_settings", all_sql)
        self.assertIn("model_definitions", all_sql)

    def test_update_title_record_fields_builds_update_sql(self) -> None:
        connection, cursor = _make_connection(fetchone_return=("TITLE123", "1HGCM82633A004352", 2003))
        update_title_record_fields(connection, record_id=1, fields={"make": "HONDA", "color": "RED"})
        # The first execute call is the UPDATE; the second fetches for re-validation.
        first_sql = repr(cursor.execute.call_args_list[0][0][0])
        self.assertIn("UPDATE", first_sql)
        self.assertIn("title_records", first_sql)

    def test_update_title_record_fields_ignores_unknown_columns(self) -> None:
        connection, cursor = _make_connection(fetchone_return=("T1", "1HGCM82633A004352", 2020))
        update_title_record_fields(
            connection, record_id=1, fields={"make": "FORD", "__evil": "DROP TABLE"}
        )
        first_sql = repr(cursor.execute.call_args_list[0][0][0])
        self.assertNotIn("__evil", first_sql)
        self.assertNotIn("DROP", first_sql)

    def test_update_title_record_fields_noop_for_empty_dict(self) -> None:
        connection, cursor = _make_connection()
        update_title_record_fields(connection, record_id=1, fields={})
        cursor.execute.assert_not_called()

    def test_insert_training_run_stores_record(self) -> None:
        connection, cursor = _make_connection(fetchone_return=(7,))
        run_id = insert_training_run(connection, sample_count=42, notes="test run")
        self.assertEqual(7, run_id)
        connection.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("INSERT INTO training_runs", sql)

    def test_list_training_runs_returns_all_rows(self) -> None:
        fake_rows = [{"id": 2, "exported_at": "2024-01-01", "sample_count": 10, "notes": None, "export_path": None}]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        runs = list_training_runs(connection)
        self.assertEqual(1, len(runs))
        self.assertEqual(2, runs[0]["id"])

    def test_get_annotation_queue_returns_rows(self) -> None:
        fake_rows = [
            {"id": 1, "state": "NM", "title_number": "T1", "vin": "1HGCM82633A004352",
             "vehicle_year": 2003, "make": None, "model": None, "color": None,
             "is_validated": 0, "created_at": "2024-01-01", "gt_count": 0},
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        records = get_annotation_queue(connection, limit=10, offset=0)
        self.assertEqual(1, len(records))
        self.assertEqual(0, records[0]["gt_count"])

    def test_get_training_stats_returns_expected_keys(self) -> None:
        cursor = mock.MagicMock()
        # fetchone called twice: for total count, then for coverage row
        cursor.fetchone.side_effect = [
            {"total": 5},
            {"total_records": 10, "ge1": 4, "ge5": 1, "ge10": 0},
        ]
        # fetchall called twice: for by_field, then for by_state
        cursor.fetchall.side_effect = [
            [{"field_name": "vin", "count": 3}],
            [{"state": "NM", "count": 5}],
        ]

        cm = mock.MagicMock()
        cm.__enter__ = mock.Mock(return_value=cursor)
        cm.__exit__ = mock.Mock(return_value=False)

        connection = mock.MagicMock()
        connection.cursor.return_value = cm

        stats = get_training_stats(connection)
        self.assertEqual(5, stats["total_gt_corrections"])
        self.assertIn("by_field", stats)
        self.assertIn("by_state", stats)
        self.assertIn("coverage", stats)
        self.assertEqual(10, stats["coverage"]["total_records"])


class AppSettingsTests(unittest.TestCase):
    """Tests for get_app_setting / set_app_setting."""

    def test_get_app_setting_returns_value_when_row_exists(self) -> None:
        connection, cursor = _make_connection(fetchone_return=("moondream",))
        result = get_app_setting(connection, "active_ai_model")
        self.assertEqual("moondream", result)
        sql = cursor.execute.call_args[0][0]
        self.assertIn("app_settings", sql)

    def test_get_app_setting_returns_default_when_not_found(self) -> None:
        connection, cursor = _make_connection(fetchone_explicit_none=True)
        result = get_app_setting(connection, "active_ai_model", default="llava")
        self.assertEqual("llava", result)

    def test_get_app_setting_returns_none_default_when_unset(self) -> None:
        connection, cursor = _make_connection(fetchone_explicit_none=True)
        result = get_app_setting(connection, "missing_key")
        self.assertIsNone(result)

    def test_set_app_setting_executes_upsert(self) -> None:
        connection, cursor = _make_connection()
        set_app_setting(connection, "active_ai_model", "titles-custom")
        connection.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("app_settings", sql)
        self.assertIn("ON CONFLICT", sql)
        params = cursor.execute.call_args[0][1]
        self.assertEqual("active_ai_model", params[0])
        self.assertEqual("titles-custom", params[1])


class ModelDefinitionsTests(unittest.TestCase):
    """Tests for upsert_model_definition / list_model_definitions."""

    def test_upsert_model_definition_executes_insert(self) -> None:
        connection, cursor = _make_connection()
        upsert_model_definition(
            connection,
            model_name="titles-custom",
            base_model="moondream",
            description="test model",
            corrections_used=5,
        )
        connection.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        self.assertIn("model_definitions", sql)
        self.assertIn("ON CONFLICT", sql)
        params = cursor.execute.call_args[0][1]
        self.assertEqual("titles-custom", params[0])
        self.assertEqual("moondream", params[1])
        self.assertEqual("test model", params[2])
        self.assertEqual(5, params[3])

    def test_upsert_model_definition_uses_none_description_when_absent(self) -> None:
        connection, cursor = _make_connection()
        upsert_model_definition(
            connection,
            model_name="titles-v2",
            base_model="llava",
        )
        params = cursor.execute.call_args[0][1]
        self.assertIsNone(params[2])   # description
        self.assertEqual(0, params[3]) # corrections_used default

    def test_list_model_definitions_returns_rows(self) -> None:
        fake_rows = [
            {
                "id": 1,
                "model_name": "titles-custom",
                "base_model": "moondream",
                "description": None,
                "corrections_used": 10,
                "created_at": "2024-01-01",
            }
        ]
        connection, _ = _make_connection(cursor_rows=fake_rows)
        rows = list_model_definitions(connection)
        self.assertEqual(1, len(rows))
        self.assertEqual("titles-custom", rows[0]["model_name"])


class MV7PdfTests(unittest.TestCase):
    """Tests for generate_mv7_pdf in pdf_generator."""

    def test_generate_mv7_raises_when_blank_form_missing(self) -> None:
        from pdf_generator import generate_mv7_pdf

        record = {
            "id": 1, "state": "NM", "title_number": "12345678",
            "vin": "1HGCM82633A004352", "owner_name": "DOE JOHN",
            "plant_name": "Test Yard", "dismantler_license": "DL001",
            "provider_id": "P001", "sale_date": "2024-01-15",
            "issue_date": None,
        }
        with mock.patch("pdf_generator._MV7_BLANK", "/nonexistent/blank.pdf"):
            with self.assertRaises(FileNotFoundError):
                generate_mv7_pdf(record, output_dir="/tmp")

    def test_generate_mv7_raises_for_invalid_row(self) -> None:
        from pdf_generator import generate_mv7_pdf

        record = {"id": 1, "state": "NM", "title_number": "T1", "vin": "X" * 17}
        with self.assertRaises(ValueError):
            generate_mv7_pdf(record, output_dir="/tmp", row=0)
        with self.assertRaises(ValueError):
            generate_mv7_pdf(record, output_dir="/tmp", row=31)

    def test_generate_mv7_fills_form_and_returns_path(self) -> None:
        """Integration test: fill the real blank MV-7 form and verify the output."""
        import os
        import tempfile

        from pdf_generator import generate_mv7_pdf, _MV7_BLANK

        if not os.path.exists(_MV7_BLANK):
            self.skipTest("BLANK-MV7-FORM.pdf not present in this environment")

        record = {
            "id": 999,
            "state": "NM",
            "title_number": "88776655",
            "vin": "1HGCM82633A004352",
            "owner_name": "DOE JOHN",
            "plant_name": "My Test Yard",
            "dismantler_license": "DL1234",
            "provider_id": "PROV99",
            "sale_date": "2024-06-01",
            "issue_date": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_mv7_pdf(record, output_dir=tmpdir, row=1)
            self.assertTrue(os.path.isfile(path))
            self.assertIn("mv7_999", os.path.basename(path))
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
